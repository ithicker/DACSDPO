import json as _json
import shlex as _shlex
import subprocess as _subprocess
import sys as _sys
from pathlib import Path as _Path


def _run_automatic_warmup():
    """One public entry point; use separate distributed phases for exact resume."""
    cli = [arg for arg in _sys.argv[1:] if arg != "--auto_warmup"]
    if "--enable_margin_threshold" not in cli:
        raise ValueError("Automatic warm-up requires a correction method")

    def value(flag, arguments):
        try:
            return arguments[arguments.index(flag) + 1]
        except (ValueError, IndexError):
            raise ValueError(f"Missing required {flag}") from None

    def replace_value(arguments, flag, new_value):
        arguments = list(arguments)
        if flag in arguments:
            arguments[arguments.index(flag) + 1] = str(new_value)
        else:
            arguments.extend((flag, str(new_value)))
        return arguments

    output_dir = _Path(value("--output_dir", cli)).resolve()
    warmup_dir = output_dir / "warmup"
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup = []
    skip_value = {"--margin_warmup_steps", "--resume_from_checkpoint"}
    skip_flag = {"--enable_margin_threshold", "--enable_grad_regularization",
                 "--disable_correction"}
    index = 0
    while index < len(cli):
        arg = cli[index]
        if arg in skip_flag:
            index += 1
        elif arg in skip_value:
            index += 2
        else:
            warmup.append(arg)
            index += 1
    warmup = replace_value(warmup, "--output_dir", warmup_dir)
    warmup.append("--warmup_only")

    launcher = [_sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=8", str(_Path(__file__).resolve())]
    warmup_command = launcher + warmup
    (output_dir / "warmup_argv.txt").write_text(_shlex.join(warmup_command) + "\n")
    _subprocess.run(warmup_command, check=True)

    stop = _json.loads((warmup_dir / "warmup_stop.json").read_text())
    checkpoint = _Path(stop["checkpoint"])
    if not stop["triggered"] or not (checkpoint / "trainer_state.json").is_file():
        raise RuntimeError("Patience stop or historical-minimum checkpoint is missing")
    continuation = replace_value(cli, "--output_dir", output_dir)
    continuation = replace_value(continuation, "--margin_warmup_steps", stop["best_step"])
    continuation = replace_value(continuation, "--resume_from_checkpoint", checkpoint)
    continuation_command = launcher + continuation
    (output_dir / "train_argv.txt").write_text(_shlex.join(continuation_command) + "\n")
    _subprocess.run(continuation_command, check=True)


if __name__ == "__main__" and "--auto_warmup" in _sys.argv:
    _run_automatic_warmup()
    raise SystemExit(0)

import torch
import torch.nn.functional as F
import math
import json
import random
from collections import deque
from pathlib import Path
from torch.func import functional_call

from datasets import load_dataset, concatenate_datasets
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainerCallback, set_seed
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

class PairGradient(torch.nn.Module):
    """Differentiate shadow adapter leaves before functional_call restores the policy."""

    def __init__(self, policy, pair_logps):
        super().__init__()
        self.policy = policy
        self.pair_logps = pair_logps

    def forward(self, batch, ref_ratio, flipped, beta, leaves):
        chosen, rejected = self.pair_logps(self.policy, batch)
        margin = beta * ((chosen - rejected) - ref_ratio)
        loss = -F.logsigmoid(torch.where(flipped, -margin, margin)).sum()
        return torch.autograd.grad(loss, leaves, allow_unused=True)


def sampled_pair_penalty(policy, pair_logps, batch, ref_ratio, flipped, beta,
                         sample_count, radius, seed):
    """Return a scalar with sampled ||g||² value and forward-FD gradient.

    All probes use private trainable leaves, so ZeRO-2 only reduces the final
    training gradient. Frozen weights are shared with the policy. The batch
    must use TRL's unpacked, leading-pair layout.
    """
    size = ref_ratio.numel()
    if ref_ratio.shape != (size,) or flipped.shape != (size,) or size == 0:
        raise ValueError("Expected nonempty, aligned per-pair margins and masks.")
    if sample_count < 1 or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Sample count and finite-difference radius must be positive.")
    count = min(sample_count, size)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator)[:count].tolist()
    params = {"policy." + n: p for n, p in policy.named_parameters() if p.requires_grad}
    if not params:
        raise ValueError("The finite-difference penalty requires trainable parameters.")
    probe = PairGradient(policy, pair_logps)
    device = ref_ratio.device
    devices = [device.index] if device.type == "cuda" else []
    value = torch.zeros((), dtype=torch.float32, device=device)
    directions = {}
    for i in indices:
        pair = {}
        for name, field in batch.items():
            if torch.is_tensor(field) and field.ndim and field.shape[0] == size:
                pair[name] = field[i:i + 1]
            elif isinstance(field, (list, tuple)) and len(field) == size:
                pair[name] = field[i:i + 1]
            elif torch.is_tensor(field) and field.ndim:
                raise ValueError(f"Unsupported batch layout for {name}: {field.shape}")
            else:
                pair[name] = field

        def gradients(shadow):
            # Same dropout realization at the two parameter points. Restore
            # process RNG afterwards, including the main DPO forward's RNG.
            with torch.random.fork_rng(devices=devices):
                probe_seed = (seed + 1 + i) % (2**63 - 1)
                torch.random.default_generator.manual_seed(probe_seed)
                if devices:
                    torch.cuda.default_generators[device.index].manual_seed(probe_seed)
                return functional_call(
                    probe, shadow,
                    (pair, ref_ratio[i:i + 1], flipped[i:i + 1], beta, tuple(shadow.values())),
                )

        shadow = {name: p.detach().requires_grad_() for name, p in params.items()}
        base = gradients(shadow)
        norm_sq = sum(g.detach().float().square().sum() for g in base if g is not None)
        if not torch.is_tensor(norm_sq) or not torch.isfinite(norm_sq).item():
            raise RuntimeError("Missing or nonfinite per-pair gradient.")
        value += norm_sq / count
        if norm_sq.item() == 0:
            continue
        norm = norm_sq.sqrt()
        for (name, p), g in zip(params.items(), base):
            if g is not None:
                shadow[name] = (p.detach().float() + (radius / norm) * g.float()).to(p.dtype).requires_grad_()
        perturbed = gradients(shadow)
        scale = 2 * norm / (radius * count)
        for (name, p), g, g_plus in zip(params.items(), base, perturbed):
            if g is None and g_plus is None:
                continue
            if name not in directions:
                directions[name] = torch.zeros_like(p, dtype=torch.float32)
            if g is None:
                delta = g_plus.float()
            elif g_plus is None:
                delta = -g.float()
            else:
                delta = g_plus.float() - g.float()
            directions[name].add_(delta * scale)

    # Zero forward value, prescribed first derivative. Trainer/DeepSpeed
    # applies normal mixed precision scaling, accumulation and rank averaging.
    squared_correction = value.new_zeros(())
    surrogate = value
    for name, direction in directions.items():
        p = params[name]
        squared_correction = squared_correction + direction.square().sum()
        surrogate = surrogate + ((p - p.detach()).float() * direction).sum()
    if not torch.isfinite(squared_correction).item():
        raise FloatingPointError("Nonfinite finite-difference correction.")
    return surrogate, squared_correction.sqrt(), count




class MarginThresholdDPOTrainer(DPOTrainer):
    """
    DACS-DPO trainer with current-policy selection and sampled penalty.

    Warm-up uses only DPO. After warm-up, the current policy margin is detached
    for selecting pairs. Eq. (8) uses a fixed negative tau; Eq. (9) computes
    alpha_t=1/(1+delta_t) and tau_t=alpha_t*tau. Selected pairs are reversed
    in Eq. (10). Eq. (12) is estimated with sampled first-order finite
    differences over the trainable LoRA parameters.
    """

    def __init__(
        self,
        *args,
        enable_margin_threshold=False,
        margin_warmup_steps=0,
        margin_threshold=-10.0,
        enable_grad_regularization=False,
        grad_reg_lambda=10.0,
        fd_pairs=1,
        fd_radius=0.05,
        disable_correction=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.enable_margin_threshold = bool(enable_margin_threshold)
        self.margin_warmup_steps = int(margin_warmup_steps)
        self.margin_threshold = float(margin_threshold)
        if self.enable_margin_threshold and self.margin_threshold >= 0:
            raise ValueError("The fixed threshold tau must be negative.")

        # Eq. (12): DACS-DPO adds alpha_t/lambda times the sampled penalty.
        self.enable_grad_regularization = bool(enable_grad_regularization)
        self.grad_reg_lambda = float(grad_reg_lambda)
        self.fd_pairs = int(fd_pairs)
        self.fd_radius = float(fd_radius)
        self.disable_correction = bool(disable_correction)
        self.fd_calls = 0
        if self.grad_reg_lambda <= 0:
            raise ValueError("grad_reg_lambda must be positive.")
        if self.fd_pairs < 1 or not math.isfinite(self.fd_radius) or self.fd_radius <= 0:
            raise ValueError("fd_pairs and fd_radius must be positive.")

        if self.enable_grad_regularization and not self.enable_margin_threshold:
            raise ValueError(
                "The DACS-DPO gradient regularizer requires "
                "enable_margin_threshold=True because alpha_t is produced by "
                "the adaptive noise-selection mechanism."
            )

        self._current_selection_margin = None
        # The fixed tau from Eq. (8).
        self._tau_base = None

        # 当前 batch 计算出的 tau_t(B)
        self._current_adaptive_threshold = None
        self._current_D = None
        self._current_alpha = None
        self._current_candidate_ratio = None

        if self.enable_margin_threshold:
            loss_type = getattr(self.args, "loss_type", "sigmoid")
            if isinstance(loss_type, list):
                valid_loss_type = len(loss_type) == 1 and loss_type[0] == "sigmoid"
            else:
                valid_loss_type = loss_type == "sigmoid"

            if not valid_loss_type:
                raise ValueError(
                    "MarginThresholdDPOTrainer only supports sigmoid DPO loss. "
                    "Please set --loss_type sigmoid."
                )

            if self.margin_warmup_steps <= 0:
                raise ValueError("A positive DPO-only warm-up length is required.")

    def _get_beta(self):
        beta = getattr(self, "beta", None)
        if beta is None:
            beta = getattr(self.args, "beta", 1.0)
        if isinstance(beta, (list, tuple)):
            beta = beta[0]
        return float(beta)

    def _is_main_process(self):
        if hasattr(self, "accelerator") and self.accelerator is not None:
            return self.accelerator.is_main_process
        return True

    def _maybe_all_reduce_sum(self, value):
        """
        多卡训练时同步 D_t(B)，保证所有 rank 的 adaptive tau 一致。
        """
        if not torch.is_tensor(value):
            return value

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            value = value.clone()
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)

        return value

    def _finalize_tau_base(self):
        """Initialize the fixed negative threshold in Eqs. (8)-(9)."""
        if self._tau_base is None:
            self._tau_base = torch.tensor(
                self.margin_threshold, dtype=torch.float32, device=self.model.device
            )

    @staticmethod
    def _unpack_concatenated_forward_output(output):
        if isinstance(output, dict):
            if "chosen_logps" in output and "rejected_logps" in output:
                return output["chosen_logps"], output["rejected_logps"]

            if "policy_chosen_logps" in output and "policy_rejected_logps" in output:
                return output["policy_chosen_logps"], output["policy_rejected_logps"]

            raise KeyError(
                f"Unknown concatenated_forward output keys: {list(output.keys())}"
            )

        if isinstance(output, (tuple, list)) and len(output) >= 2:
            return output[0], output[1]

        raise TypeError(
            f"Unsupported concatenated_forward output type: {type(output)}"
        )

    def _get_pad_token_id(self):
        tokenizer = getattr(self, "processing_class", None)

        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
            return tokenizer.pad_token_id

        tokenizer = getattr(self, "tokenizer", None)

        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
            return tokenizer.pad_token_id

        return 0

    def _get_response_lengths(self, batch, reference_chosen_logps):
        """
        计算 chosen/rejected response 长度，用于 reference logp 长度归一化。
        """
        device = reference_chosen_logps.device

        if "chosen_labels" in batch and "rejected_labels" in batch:
            chosen_len = (batch["chosen_labels"] != -100).sum(dim=-1)
            rejected_len = (batch["rejected_labels"] != -100).sum(dim=-1)

        elif "chosen_attention_mask" in batch and "rejected_attention_mask" in batch:
            chosen_len = batch["chosen_attention_mask"].sum(dim=-1)
            rejected_len = batch["rejected_attention_mask"].sum(dim=-1)

        elif "chosen_input_ids" in batch and "rejected_input_ids" in batch:
            pad_token_id = self._get_pad_token_id()
            chosen_len = (batch["chosen_input_ids"] != pad_token_id).sum(dim=-1)
            rejected_len = (batch["rejected_input_ids"] != pad_token_id).sum(dim=-1)

        else:
            chosen_len = torch.ones_like(reference_chosen_logps)
            rejected_len = torch.ones_like(reference_chosen_logps)

        chosen_len = chosen_len.to(device=device, dtype=reference_chosen_logps.dtype)
        rejected_len = rejected_len.to(device=device, dtype=reference_chosen_logps.dtype)

        chosen_len = torch.clamp(chosen_len, min=1.0)
        rejected_len = torch.clamp(rejected_len, min=1.0)

        return chosen_len, rejected_len

    def _compute_adaptive_threshold(
        self, batch, selection_margin, reference_chosen_logps,
        reference_rejected_logps,
    ):
        """Eqs. (8)-(9), synchronized over the full distributed mini-batch."""
        if self._tau_base is None:
            raise RuntimeError("Fixed tau was not initialized after warm-up.")
        with torch.no_grad():
            tau = self._tau_base.to(selection_margin.device)
            chosen_len, rejected_len = self._get_response_lengths(
                batch, reference_chosen_logps
            )
            distinguishability = (
                reference_chosen_logps.float() / chosen_len.float()
                - reference_rejected_logps.float() / rejected_len.float()
            ).abs()
            candidates = selection_margin.detach() < tau
            delta_t = self._maybe_all_reduce_sum(
                (candidates.float() * distinguishability).sum()
            )
            candidate_count = self._maybe_all_reduce_sum(candidates.float().sum())
            total_count = self._maybe_all_reduce_sum(
                candidates.new_tensor(candidates.numel(), dtype=torch.float32)
            )
            alpha_t = 1.0 / (1.0 + delta_t)
            tau_t = alpha_t * tau
            candidate_ratio = candidate_count / total_count.clamp_min(1.0)
        return (tau_t.detach(), delta_t.detach(), alpha_t.detach(),
                candidate_ratio.detach(), tau.detach())

    def dpo_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        *loss_args,
        **loss_kwargs,
    ):
        beta = self._get_beta()

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        student_logits = pi_logratios - ref_logratios
        student_margin = beta * student_logits

        chosen_rewards = beta * (
            policy_chosen_logps - reference_chosen_logps
        ).detach()

        rejected_rewards = beta * (
            policy_rejected_logps - reference_rejected_logps
        ).detach()

        normal_losses = -F.logsigmoid(student_margin)
        flipped_losses = -F.logsigmoid(-student_margin)

        current_step = getattr(self.state, "global_step", 0)

        if (
            not self.enable_margin_threshold
            or current_step < self.margin_warmup_steps
            or not self.model.training
        ):
            return normal_losses, chosen_rewards, rejected_rewards

        if self._current_selection_margin is None:
            raise RuntimeError(
                "Current-policy selection margin is unavailable."
            )

        if self._current_adaptive_threshold is None:
            raise RuntimeError(
                "Adaptive threshold is None. "
                "get_batch_loss_metrics must compute adaptive threshold before dpo_loss."
            )

        with torch.no_grad():
            selection_margin = self._current_selection_margin.detach().to(normal_losses.device)
            adaptive_threshold = self._current_adaptive_threshold.detach().to(normal_losses.device)

            if selection_margin.shape != normal_losses.shape:
                raise RuntimeError(
                    f"selection_margin shape {selection_margin.shape} does not match "
                    f"normal_losses shape {normal_losses.shape}"
                )

            noisy_mask = selection_margin < adaptive_threshold

        losses = (normal_losses if self.disable_correction else
                  torch.where(noisy_mask, flipped_losses, normal_losses))

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        current_step = getattr(self.state, "global_step", 0)

        policy_output = self.concatenated_forward(model, batch)
        policy_chosen_logps, policy_rejected_logps = (
            self._unpack_concatenated_forward_output(policy_output)
        )

        with torch.no_grad():
            if getattr(self, "ref_model", None) is None:
                with self.null_ref_context():
                    ref_output = self.concatenated_forward(model, batch)
            else:
                ref_output = self.concatenated_forward(self.ref_model, batch)

        reference_chosen_logps, reference_rejected_logps = (
            self._unpack_concatenated_forward_output(ref_output)
        )

        self._current_selection_margin = None
        self._current_adaptive_threshold = None
        self._current_D = None
        self._current_alpha = None
        self._current_candidate_ratio = None
        self._current_tau_base_for_log = None

        if (
            self.enable_margin_threshold
            and self.model.training
            and current_step >= self.margin_warmup_steps
        ):
            self._finalize_tau_base()
            beta = self._get_beta()

            with torch.no_grad():
                # Freeze this batch's current policy margin for selection.
                selection_pi_logratios = (
                    policy_chosen_logps.detach() - policy_rejected_logps.detach()
                )
                ref_logratios = reference_chosen_logps - reference_rejected_logps
                selection_logits = selection_pi_logratios - ref_logratios
                self._current_selection_margin = beta * selection_logits

                (
                    tau_t,
                    D_t,
                    alpha_t,
                    candidate_ratio,
                    tau_base,
                ) = self._compute_adaptive_threshold(
                    batch=batch,
                    selection_margin=self._current_selection_margin,
                    reference_chosen_logps=reference_chosen_logps,
                    reference_rejected_logps=reference_rejected_logps,
                )

                self._current_adaptive_threshold = tau_t
                self._current_D = D_t
                self._current_alpha = alpha_t
                self._current_candidate_ratio = candidate_ratio
                self._current_tau_base_for_log = tau_base

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        # The original full-batch DPO/DAC loss remains unchanged in scale.
        dac_dpo_loss_mean = losses.mean()
        total_loss = dac_dpo_loss_mean

        grad_norm_penalty = None
        grad_reg_weight = None

        # Eq. (12): sampled per-pair finite-difference gradient penalty.
        # No second-order attention backward is requested.
        if (
            self.enable_grad_regularization
            and train_eval == "train"
            and model.training
            and torch.is_grad_enabled()
            and current_step >= self.margin_warmup_steps
            and self._current_alpha is not None
        ):
            raw_model = self.accelerator.unwrap_model(model)
            flipped = (
                self._current_selection_margin < self._current_adaptive_threshold
            ).detach()
            if self.disable_correction:
                flipped = torch.zeros_like(flipped)

            def pair_logps(policy, pair):
                return self._unpack_concatenated_forward_output(
                    self.concatenated_forward(policy, pair)
                )

            sample_seed = (self.args.seed + 104729 * self.fd_calls
                           + 1000000007 * self.accelerator.process_index)
            grad_norm_penalty, _, _ = sampled_pair_penalty(
                raw_model, pair_logps, batch,
                (reference_chosen_logps - reference_rejected_logps).detach(),
                flipped, self._get_beta(), self.fd_pairs, self.fd_radius,
                sample_seed,
            )
            self.fd_calls += 1

            # alpha_t is a batch/step-level adaptive scalar produced by the
            # noise-selection module. It is detached from theta by design.
            alpha_t = self._current_alpha.detach().to(
                device=grad_norm_penalty.device,
                dtype=grad_norm_penalty.dtype,
            )
            grad_reg_weight = alpha_t / self.grad_reg_lambda

            total_loss = (
                dac_dpo_loss_mean
                + grad_reg_weight * grad_norm_penalty
            )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margins = chosen_rewards - rejected_rewards

        prefix = "eval_" if train_eval == "eval" else ""

        metrics = {}
        metrics[f"{prefix}loss/dac_dpo"] = (
            dac_dpo_loss_mean.detach().float().cpu()
        )

        if grad_norm_penalty is not None:
            weighted_penalty = grad_reg_weight * grad_norm_penalty
            metrics["grad_reg/alpha_t"] = self._current_alpha.detach().float().cpu()
            metrics["grad_reg/lambda"] = torch.tensor(self.grad_reg_lambda)
            metrics["grad_reg/weight"] = grad_reg_weight.detach().float().cpu()
            metrics["grad_reg/penalty"] = grad_norm_penalty.detach().float().cpu()
            metrics["grad_reg/weighted_penalty"] = weighted_penalty.detach().float().cpu()
            metrics["loss/total"] = total_loss.detach().float().cpu()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().detach().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().detach().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().detach().cpu()
        metrics[f"{prefix}rewards/margins"] = reward_margins.mean().detach().cpu()

        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().detach().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().detach().cpu()

        if (
            self.enable_margin_threshold
            and self.model.training
            and self._current_selection_margin is not None
            and self._current_adaptive_threshold is not None
        ):
            with torch.no_grad():
                selection_margin = self._current_selection_margin.detach()
                final_noisy_ratio = (
                    selection_margin < self._current_adaptive_threshold.to(selection_margin.device)
                ).float().mean()

            metrics[f"{prefix}selection_margin/mean"] = selection_margin.mean().detach().cpu()
            metrics[f"{prefix}selection_margin/min"] = selection_margin.min().detach().cpu()
            metrics[f"{prefix}selection_margin/max"] = selection_margin.max().detach().cpu()

            metrics[f"{prefix}adaptive_threshold/tau_base"] = self._current_tau_base_for_log.detach().cpu()
            metrics[f"{prefix}adaptive_threshold/tau_t"] = self._current_adaptive_threshold.detach().cpu()
            metrics[f"{prefix}adaptive_threshold/alpha"] = self._current_alpha.detach().cpu()
            metrics[f"{prefix}adaptive_threshold/D"] = self._current_D.detach().cpu()
            metrics[f"{prefix}adaptive_threshold/candidate_ratio"] = self._current_candidate_ratio.detach().cpu()
            metrics[f"{prefix}adaptive_threshold/final_noisy_ratio"] = final_noisy_ratio.detach().cpu()

        return total_loss, metrics

class PatienceWarmup(TrainerCallback):
    """Algorithm 1: save the full checkpoint at each new probe DC minimum."""

    def __init__(self, trainer, probe_size, window, patience, epsilon, seed):
        self.trainer = trainer
        self.window = window
        self.patience = patience
        self.epsilon = epsilon
        self.indices = random.Random(seed).sample(
            range(len(trainer.train_dataset)), probe_size
        )
        self.probe = trainer.train_dataset.select(self.indices)
        self.history = deque(maxlen=window)
        self.previous = None
        self.best_value = float("inf")
        self.best_step = None
        self.bad_count = 0
        self.confirm_step = None
        self.trace = []

    @staticmethod
    def _logps(output):
        if isinstance(output, dict):
            return output["chosen_logps"], output["rejected_logps"]
        return output[0], output[1]

    def _probe_margins(self):
        trainer = self.trainer
        model = trainer.model_wrapped or trainer.model
        was_training = model.training
        model.eval()
        margins = []
        try:
            with torch.no_grad():
                for start in range(0, len(self.probe), 4):
                    features = [self.probe[i] for i in range(start, min(start + 4, len(self.probe)))]
                    batch = trainer._prepare_inputs(trainer.data_collator(features))
                    chosen, rejected = self._logps(trainer.concatenated_forward(model, batch))
                    if trainer.ref_model is None:
                        with trainer.null_ref_context():
                            ref_chosen, ref_rejected = self._logps(
                                trainer.concatenated_forward(model, batch)
                            )
                    else:
                        ref_chosen, ref_rejected = self._logps(
                            trainer.concatenated_forward(trainer.ref_model, batch)
                        )
                    margins.append((trainer.beta * (
                        chosen.float() - rejected.float()
                        - ref_chosen.float() + ref_rejected.float()
                    )).detach())
                result = torch.cat(margins)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
                    result /= torch.distributed.get_world_size()
                return result
        finally:
            model.train(was_training)

    def on_train_begin(self, args, state, control, **kwargs):
        self.previous = self._probe_margins()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        current = self._probe_margins()
        change = (current - self.previous).abs().div(self.previous.abs() + self.epsilon).mean()
        self.previous = current
        self.history.append(float(change.item()))
        if len(self.history) < self.window:
            return control
        dc = sum(self.history) / self.window
        if not math.isfinite(dc):
            raise FloatingPointError("Nonfinite warm-up DC value")
        if dc <= self.best_value:
            self.best_value = dc
            self.best_step = state.global_step
            self.bad_count = 0
            control.should_save = True
        else:
            self.bad_count += 1
        self.trace.append({"step": state.global_step, "dc": dc,
                           "best_step": self.best_step, "bad_count": self.bad_count})
        if self.bad_count >= self.patience:
            self.confirm_step = state.global_step
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint = (Path(args.output_dir) / f"checkpoint-{self.best_step}").resolve()
            result = {"best_step": self.best_step, "confirm_step": self.confirm_step,
                      "best_dc": self.best_value, "patience": self.patience,
                      "window": self.window, "probe_indices": self.indices,
                      "checkpoint": str(checkpoint), "triggered": self.confirm_step is not None}
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.output_dir) / "warmup_stop.json").write_text(json.dumps(result, indent=2))
            (Path(args.output_dir) / "warmup_dc.json").write_text(json.dumps(self.trace, indent=2))
        return control


METHOD_TRAINERS = {
    "DPO": (DPOTrainer, DPOConfig),
    "CPO": (CPOTrainer, CPOConfig),
    "KTO": (KTOTrainer, KTOConfig),
    "ORPO": (ORPOTrainer, ORPOConfig),
}


pipeline_name = "finetuner"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))

parser.add_argument(
    "--opt_alg",
    type=str,
    choices=["DPO", "CPO", "KTO", "ORPO"],
    required=True,
)
parser.add_argument("--loss_type", default="sigmoid")
parser.add_argument("--beta", type=float, default=0.1)

parser.add_argument(
    "--enable_margin_threshold",
    action="store_true",
)
parser.add_argument(
    "--margin_warmup_steps",
    type=int,
    default=0,
)
parser.add_argument(
    "--margin_threshold",
    type=float,
    default=-10.0,
)
parser.add_argument(
    "--enable_grad_regularization",
    action="store_true",
)
parser.add_argument(
    "--grad_reg_lambda",
    type=float,
    default=10.0,
    help="Positive lambda in Eq. (12); penalty weight is alpha_t/lambda.",
)
parser.add_argument("--fd_pairs", type=int, default=1)
parser.add_argument("--fd_radius", type=float, default=0.05)
parser.add_argument("--disable_correction", action="store_true")
parser.add_argument("--warmup_only", action="store_true")
parser.add_argument("--warmup_probe_size", type=int, default=20)
parser.add_argument("--warmup_window", type=int, default=10)
parser.add_argument("--warmup_patience", type=int, default=50)
parser.add_argument("--warmup_epsilon", type=float, default=1e-6)
parser.add_argument("--resume_from_checkpoint", type=str, default="")

args = parser.parse_args_into_dataclasses()
model_args, data_args, pipeline_args = args[:3]
method_args = parser.parse_args()
set_seed(pipeline_args.seed)

print("^^^^^^**^^^^^^^^")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    load_in_8bit=True,
)
tokenizer.pad_token = tokenizer.eos_token


print("Loading model...")
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    torch_dtype=torch.bfloat16,
)

print(model_args.model_name_or_path)
print(model_args.use_lora)

if model_args.use_lora:
    model_lora = get_peft_model(model, peft_config)
    model_lora.print_trainable_parameters()
else:
    model_lora = model

model_lora.config.pad_token_id = tokenizer.eos_token_id


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
    ds = load_dataset(
        "json",
        data_files=f"{config.dataset_path}/train.jsonl",
    )["train"]

    test_path = "/".join(config.dataset_path.split("/")[:-1])
    test_ds = load_dataset(
        "json",
        data_files=f"{test_path}/no_clean/test.jsonl",
    )["train"]

    if opt_alg == "KTO":
        train_dataset = concatenate_datasets(
            [
                ds.map(extract_positive),
                ds.map(extract_negative),
            ]
        )
        eval_dataset = concatenate_datasets(
            [
                test_ds.map(extract_positive),
                test_ds.map(extract_negative),
            ]
        )
    else:
        train_dataset, eval_dataset = ds, test_ds

    return train_dataset, eval_dataset


train_dataset, eval_dataset = build_dataset(data_args, method_args.opt_alg)
print(f"Training samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")


trainer_class, config_class = METHOD_TRAINERS[method_args.opt_alg]

if method_args.opt_alg == "DPO" and (
    method_args.enable_margin_threshold
    or method_args.enable_grad_regularization
):
    trainer_class = MarginThresholdDPOTrainer

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
    max_steps=pipeline_args.max_steps,
    bf16=pipeline_args.bf16,
    seed=pipeline_args.seed,
    do_train=pipeline_args.do_train,
    do_eval=pipeline_args.do_eval,
    run_name=pipeline_args.run_name,
    eval_strategy=pipeline_args.eval_strategy,
    eval_steps=pipeline_args.eval_steps,
    save_steps=pipeline_args.save_steps,
    save_strategy="steps",
    save_only_model=False,
    deepspeed=pipeline_args.deepspeed,
    logging_steps=pipeline_args.logging_steps,
)

if method_args.opt_alg == "DPO":
    if isinstance(getattr(trainer_args, "loss_type", None), list):
        trainer_args.loss_type = [method_args.loss_type]
    else:
        trainer_args.loss_type = method_args.loss_type


trainer_kwargs = dict(
    model=model_lora,
    args=trainer_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

if trainer_class is MarginThresholdDPOTrainer:
    trainer_kwargs.update(
        dict(
            enable_margin_threshold=method_args.enable_margin_threshold,
            margin_warmup_steps=method_args.margin_warmup_steps,
            margin_threshold=method_args.margin_threshold,
            enable_grad_regularization=method_args.enable_grad_regularization,
            grad_reg_lambda=method_args.grad_reg_lambda,
            fd_pairs=method_args.fd_pairs,
            fd_radius=method_args.fd_radius,
            disable_correction=method_args.disable_correction,
        )
    )


trainer = trainer_class(**trainer_kwargs)

if method_args.warmup_only:
    if method_args.opt_alg != "DPO" or method_args.enable_margin_threshold:
        raise ValueError("Warm-up phase must optimize ordinary DPO only")
    if (method_args.warmup_probe_size < 1
            or method_args.warmup_probe_size > len(trainer.train_dataset)
            or method_args.warmup_window < 1
            or method_args.warmup_patience < 1
            or method_args.warmup_epsilon <= 0):
        raise ValueError("Invalid probe size, window, patience, or epsilon")
    callback = PatienceWarmup(
        trainer, method_args.warmup_probe_size, method_args.warmup_window,
        method_args.warmup_patience, method_args.warmup_epsilon, pipeline_args.seed
    )
    trainer.add_callback(callback)
    trainer.train()
    if callback.confirm_step is None:
        raise RuntimeError("Patience did not trigger within the training budget")
    if not (Path(pipeline_args.output_dir) / f"checkpoint-{callback.best_step}" / "trainer_state.json").is_file():
        raise RuntimeError("Historical-minimum checkpoint was not saved")
    raise SystemExit(0)

if method_args.enable_margin_threshold and not method_args.resume_from_checkpoint:
    raise ValueError("Correction requires the patience-selected full checkpoint")
trainer.train(resume_from_checkpoint=method_args.resume_from_checkpoint or None)

if model_args.use_lora:
    model_lora = model_lora.merge_and_unload()

model_lora.save_pretrained(
    pipeline_args.output_dir,
    safe_serialization=False,
)
model.config.save_pretrained(pipeline_args.output_dir)
tokenizer.save_pretrained(pipeline_args.output_dir)
