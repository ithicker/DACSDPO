#flip & choose

import torch

from datasets import load_dataset, concatenate_datasets

from lmflow.args import ModelArguments, DatasetArguments, AutoArguments

from peft import LoraConfig, TaskType, get_peft_model

from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import (
    DPOConfig,
    DPOTrainer,
    CPOConfig,
    CPOTrainer,
    KTOConfig,
    KTOTrainer,
    ORPOConfig,
    ORPOTrainer,
)


# ==========================================================
# Trainer mapping
# ==========================================================

METHOD_TRAINERS = {
    "DPO": (DPOTrainer, DPOConfig),
    "CPO": (CPOTrainer, CPOConfig),
    "KTO": (KTOTrainer, KTOConfig),
    "ORPO": (ORPOTrainer, ORPOConfig),
}


# ==========================================================
# Batch-level margin DPO trainer
# ==========================================================

class BatchMarginDPOTrainer(DPOTrainer):
    """
    Batch-level online choose / flip / drop with warmup.

    global_step < margin_warmup_steps:
        standard DPO loss.

    global_step >= margin_warmup_steps:
        compute DPO implicit reward margin:

            reward_chosen =
                beta * (policy_chosen_logp - ref_chosen_logp)

            reward_rejected =
                beta * (policy_rejected_logp - ref_rejected_logp)

            margin =
                reward_chosen - reward_rejected

        equivalently:

            margin =
                beta * (
                    (policy_chosen_logp - policy_rejected_logp)
                    - (ref_chosen_logp - ref_rejected_logp)
                )

        margin >= tau_pos:
            clean, use DPO(chosen, rejected), weight = 1

        margin <= tau_neg:
            flip, use DPO(rejected, chosen), weight = 1

        tau_neg < margin < tau_pos:
            uncertain, weight = 0
    """

    def __init__(
        self,
        *args,
        margin_warmup_steps=500,
        tau_pos=0.2,
        tau_neg=-0.2,
        normalize_active_loss=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.margin_warmup_steps = margin_warmup_steps
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.normalize_active_loss = normalize_active_loss

        self._last_batch_margin_stats = None

    def _compute_reward_margin(
        self,
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    ):
        reward_chosen = self.beta * (
            chosen_logps - ref_chosen_logps
        )

        reward_rejected = self.beta * (
            rejected_logps - ref_rejected_logps
        )

        margin = reward_chosen - reward_rejected

        return margin, reward_chosen, reward_rejected

    def dpo_loss(
        self,
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    ):
        current_step = int(self.state.global_step)

        # --------------------------------------------------
        # 1. Warmup stage: standard DPO
        # --------------------------------------------------

        if current_step < self.margin_warmup_steps:
            losses, chosen_rewards, rejected_rewards = super().dpo_loss(
                chosen_logps,
                rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )

            with torch.no_grad():
                margin, reward_chosen, reward_rejected = self._compute_reward_margin(
                    chosen_logps=chosen_logps,
                    rejected_logps=rejected_logps,
                    ref_chosen_logps=ref_chosen_logps,
                    ref_rejected_logps=ref_rejected_logps,
                )

                policy_margin = chosen_logps - rejected_logps
                ref_margin = ref_chosen_logps - ref_rejected_logps

                self._last_batch_margin_stats = {
                    "mode": "warmup",
                    "clean_ratio": torch.tensor(1.0, device=chosen_logps.device),
                    "flip_ratio": torch.tensor(0.0, device=chosen_logps.device),
                    "drop_ratio": torch.tensor(0.0, device=chosen_logps.device),
                    "active_ratio": torch.tensor(1.0, device=chosen_logps.device),
                    "margin_mean": margin.mean().detach(),
                    "reward_chosen_mean": reward_chosen.mean().detach(),
                    "reward_rejected_mean": reward_rejected.mean().detach(),
                    "policy_margin_mean": policy_margin.mean().detach(),
                    "ref_margin_mean": ref_margin.mean().detach(),
                }

            return losses, chosen_rewards, rejected_rewards

        # --------------------------------------------------
        # 2. After warmup: batch-level choose / flip / drop
        # --------------------------------------------------

        with torch.no_grad():
            margin, reward_chosen, reward_rejected = self._compute_reward_margin(
                chosen_logps=chosen_logps,
                rejected_logps=rejected_logps,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
            )

            clean_mask = margin >= self.tau_pos
            flip_mask = margin <= self.tau_neg
            active_mask = clean_mask | flip_mask

            weights = active_mask.to(dtype=chosen_logps.dtype)

        # --------------------------------------------------
        # 3. Effective chosen / rejected log-probs
        # --------------------------------------------------

        effective_chosen_logps = torch.where(
            flip_mask,
            rejected_logps,
            chosen_logps,
        )

        effective_rejected_logps = torch.where(
            flip_mask,
            chosen_logps,
            rejected_logps,
        )

        effective_ref_chosen_logps = torch.where(
            flip_mask,
            ref_rejected_logps,
            ref_chosen_logps,
        )

        effective_ref_rejected_logps = torch.where(
            flip_mask,
            ref_chosen_logps,
            ref_rejected_logps,
        )

        # --------------------------------------------------
        # 4. Standard DPO loss on effective pairs
        # --------------------------------------------------

        losses, chosen_rewards, rejected_rewards = super().dpo_loss(
            effective_chosen_logps,
            effective_rejected_logps,
            effective_ref_chosen_logps,
            effective_ref_rejected_logps,
        )

        # --------------------------------------------------
        # 5. Drop uncertain samples
        # --------------------------------------------------

        losses = losses * weights

        if self.normalize_active_loss:
            active_count = weights.sum().clamp_min(1.0)

            batch_size = torch.tensor(
                weights.numel(),
                dtype=weights.dtype,
                device=weights.device,
            )

            losses = losses * batch_size / active_count

        # --------------------------------------------------
        # 6. Logging stats
        # --------------------------------------------------

        with torch.no_grad():
            policy_margin = chosen_logps - rejected_logps
            ref_margin = ref_chosen_logps - ref_rejected_logps

            self._last_batch_margin_stats = {
                "mode": "batch_margin_filter",
                "clean_ratio": clean_mask.float().mean().detach(),
                "flip_ratio": flip_mask.float().mean().detach(),
                "drop_ratio": (~active_mask).float().mean().detach(),
                "active_ratio": active_mask.float().mean().detach(),
                "margin_mean": margin.mean().detach(),
                "reward_chosen_mean": reward_chosen.mean().detach(),
                "reward_rejected_mean": reward_rejected.mean().detach(),
                "policy_margin_mean": policy_margin.mean().detach(),
                "ref_margin_mean": ref_margin.mean().detach(),
            }

        return losses, chosen_rewards, rejected_rewards

    def log(self, logs, *args, **kwargs):
        if self._last_batch_margin_stats is not None:
            for key, value in self._last_batch_margin_stats.items():
                if key == "mode":
                    continue

                if torch.is_tensor(value):
                    logs[f"batch_margin/{key}"] = value.item()
                else:
                    logs[f"batch_margin/{key}"] = value

        return super().log(logs, *args, **kwargs)


# ==========================================================
# Dataset helpers
# ==========================================================

def extract_positive(sample):
    return {
        "prompt": sample["prompt"],
        "completion": sample["chosen"],
        "label": True,
    }


def extract_negative(sample):
    return {
        "prompt": sample["prompt"],
        "completion": sample["rejected"],
        "label": False,
    }


def build_dataset(config, opt_alg):
    train_ds = load_dataset(
        "json",
        data_files=f"{config.dataset_path}/train.jsonl",
    )["train"]

    test_path = "/".join(config.dataset_path.split("/")[:-1])

    test_ds = load_dataset(
        "json",
        data_files=f"{test_path}/no_clean/test.jsonl",
    )["train"]

    # 原代码这里写的是 opt_alg == 'kto'。
    # 由于 opt_alg choices 是大写 KTO，这里修正为 KTO。
    if opt_alg == "KTO":
        train_dataset = concatenate_datasets(
            [
                train_ds.map(extract_positive),
                train_ds.map(extract_negative),
            ]
        )

        eval_dataset = concatenate_datasets(
            [
                test_ds.map(extract_positive),
                test_ds.map(extract_negative),
            ]
        )

    else:
        train_dataset = train_ds
        eval_dataset = test_ds

    return train_dataset, eval_dataset


# ==========================================================
# Main
# ==========================================================

def main():
    # ------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------

    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser(
        (
            ModelArguments,
            DatasetArguments,
            PipelineArguments,
        )
    )

    parser.add_argument(
        "--opt_alg",
        type=str,
        choices=["DPO", "CPO", "KTO", "ORPO"],
        required=True,
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
    )

    # ------------------------------------------------------
    # New optional arguments for our method
    # ------------------------------------------------------
    # Do not use --warmup_steps here.
    # Some existing PipelineArguments already define --warmup_steps.

    parser.add_argument(
        "--enable_batch_margin_filter",
        action="store_true",
    )

    parser.add_argument(
        "--margin_warmup_steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--tau_pos",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--tau_neg",
        type=float,
        default=-0.2,
    )

    parser.add_argument(
        "--normalize_active_loss",
        action="store_true",
    )

    parsed_args = parser.parse_args_into_dataclasses()

    model_args = parsed_args[0]
    data_args = parsed_args[1]
    pipeline_args = parsed_args[2]
    method_args = parsed_args[3]

    print(model_args)
    print(data_args)
    print(pipeline_args)
    print(method_args)

    # ------------------------------------------------------
    # Setup tokenizer
    # ------------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=True,
    )

    tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------
    # Setup model
    # ------------------------------------------------------

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )

    print(model_args.model_name_or_path)
    print(model_args.use_lora)

    if model_args.use_lora:
        # Original code used LoRA r=16, alpha=32, dropout=0.1.
        # target_modules is added because the current PEFT version requires it.
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        model_lora = get_peft_model(
            model,
            peft_config,
        )

        model_lora.print_trainable_parameters()

    else:
        model_lora = model

    model_lora.config.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------

    train_dataset, eval_dataset = build_dataset(
        data_args,
        method_args.opt_alg,
    )

    print(
        f"Training samples: {len(train_dataset)}, "
        f"Eval samples: {len(eval_dataset)}"
    )

    # ------------------------------------------------------
    # Setup config and trainer
    # ------------------------------------------------------

    trainer_class, config_class = METHOD_TRAINERS[method_args.opt_alg]

    trainer_args = config_class(
        output_dir=pipeline_args.output_dir,
        beta=method_args.beta,
        max_length=1024,
        max_prompt_length=512,
        learning_rate=pipeline_args.learning_rate,
        weight_decay=pipeline_args.weight_decay,
        lr_scheduler_type=pipeline_args.lr_scheduler_type,
        per_device_train_batch_size=pipeline_args.per_device_train_batch_size,
        per_device_eval_batch_size=pipeline_args.per_device_eval_batch_size,
        num_train_epochs=pipeline_args.num_train_epochs,
        bf16=pipeline_args.bf16,
        seed=pipeline_args.seed,
        do_train=pipeline_args.do_train,
        do_eval=pipeline_args.do_eval,
        run_name=pipeline_args.run_name,
        eval_strategy=pipeline_args.eval_strategy,
        eval_steps=pipeline_args.eval_steps,
        save_steps=pipeline_args.save_steps,
        deepspeed=pipeline_args.deepspeed,
        logging_steps = pipeline_args.logging_steps,
    )

    if method_args.opt_alg == "DPO":
        trainer_args.loss_type = method_args.loss_type

    # ------------------------------------------------------
    # Build trainer
    # ------------------------------------------------------

    if method_args.enable_batch_margin_filter:
        if method_args.opt_alg != "DPO":
            raise ValueError(
                "--enable_batch_margin_filter is currently implemented only for DPO."
            )

        trainer = BatchMarginDPOTrainer(
            model=model_lora,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            margin_warmup_steps=method_args.margin_warmup_steps,
            tau_pos=method_args.tau_pos,
            tau_neg=method_args.tau_neg,
            normalize_active_loss=method_args.normalize_active_loss,
        )

        print("Using BatchMarginDPOTrainer.")
        print(f"margin_warmup_steps = {method_args.margin_warmup_steps}")
        print(f"tau_pos = {method_args.tau_pos}")
        print(f"tau_neg = {method_args.tau_neg}")
        print(f"normalize_active_loss = {method_args.normalize_active_loss}")

    else:
        trainer = trainer_class(
            model=model_lora,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    trainer.train()

    # ------------------------------------------------------
    # Save final model
    # ------------------------------------------------------

    if model_args.use_lora:
        model_lora = model_lora.merge_and_unload()

    model_lora.save_pretrained(
        pipeline_args.output_dir,
        safe_serialization=False,
    )

    model.config.save_pretrained(
        pipeline_args.output_dir,
    )

    tokenizer.save_pretrained(
        pipeline_args.output_dir,
    )


if __name__ == "__main__":
    main()