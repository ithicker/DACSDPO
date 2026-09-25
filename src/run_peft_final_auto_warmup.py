from collections import deque

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

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

class MarginThresholdDPOTrainer(DPOTrainer):
    """
    Margin-guided Noise Correction with Stability DPO (MNCS-DPO).

    核心逻辑:
        1. warmup 阶段只使用标准 DPO。
        2. 自动 warmup 模式在固定 probe set D_m 上定期计算 margin 向量 M_t，
           再由 MC_t、移动平均 MC_bar_t 和 patience 自动确定停止点。
        3. 将历史最优 MC_bar_t 对应的模型保存为 frozen teacher；student 不回退。
        4. 同步保存该最优时刻之前累计的 warmup 最小 margin，用于初始化 tau_base。
        5. warmup 后每个 batch：
            - teacher 计算 teacher_margin；
            - 用 tau_prev 筛选候选 noisy pair 并计算 D_t(B)；
            - alpha_t(B) = 1 - sigmoid(D_t(B))；
            - tau_t(B) = alpha_t(B) * tau_base；
            - teacher_margin <= tau_t(B) 的样本使用 flipped DPO loss；
            - 其余样本使用 normal DPO loss。
        6. 最终使用动态稳定性正则：
           L = L_MNC-DPO + (alpha_t / lambda) * ||grad L_MNC-DPO||_2^2。
    """

    def __init__(
        self,
        *args,
        enable_margin_threshold=False,
        margin_warmup_steps=0,
        margin_threshold=-10.0,
        enable_grad_regularization=False,
        grad_reg_lambda=10.0,
        enable_auto_warmup=False,
        warmup_probe_size=256,
        warmup_probe_batch_size=0,
        warmup_probe_interval=200,
        warmup_ma_window=5,
        warmup_patience=3,
        warmup_max_steps=5000,
        warmup_mc_eps=1e-6,
        warmup_mc_min_delta=0.0,
        warmup_probe_seed=42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.enable_margin_threshold = bool(enable_margin_threshold)
        self.margin_warmup_steps = int(margin_warmup_steps)
        self.margin_threshold = float(margin_threshold)

        # Dynamic gradient-norm regularization for MNCS-DPO:
        #   L_total = L_MNC-DPO
        #             + (alpha_t / lambda) * ||grad_theta L_MNC-DPO||_2^2
        self.enable_grad_regularization = bool(enable_grad_regularization)
        self.grad_reg_lambda = float(grad_reg_lambda)
        if self.grad_reg_lambda <= 0:
            raise ValueError("grad_reg_lambda must be positive.")

        if self.enable_grad_regularization and not self.enable_margin_threshold:
            raise ValueError(
                "The MNCS-DPO gradient regularizer requires "
                "enable_margin_threshold=True because alpha_t is produced by "
                "the adaptive noise-selection mechanism."
            )

        # --------------------------------------------------------------
        # Automatic warm-up configuration.
        # --------------------------------------------------------------
        self.enable_auto_warmup = bool(enable_auto_warmup)
        self.warmup_probe_size = int(warmup_probe_size)
        self.warmup_probe_batch_size = int(warmup_probe_batch_size)
        self.warmup_probe_interval = int(warmup_probe_interval)
        self.warmup_ma_window = int(warmup_ma_window)
        self.warmup_patience = int(warmup_patience)
        self.warmup_max_steps = int(warmup_max_steps)
        self.warmup_mc_eps = float(warmup_mc_eps)
        self.warmup_mc_min_delta = float(warmup_mc_min_delta)
        self.warmup_probe_seed = int(warmup_probe_seed)

        if self.enable_auto_warmup and not self.enable_margin_threshold:
            raise ValueError(
                "Automatic warm-up is part of the margin-guided noise-selection "
                "pipeline and therefore requires enable_margin_threshold=True."
            )

        if self.enable_auto_warmup:
            if self.warmup_probe_size <= 0:
                raise ValueError("warmup_probe_size must be positive.")
            if self.warmup_probe_batch_size < 0:
                raise ValueError("warmup_probe_batch_size must be >= 0.")
            if self.warmup_probe_interval <= 0:
                raise ValueError("warmup_probe_interval must be positive.")
            if self.warmup_ma_window <= 0:
                raise ValueError("warmup_ma_window must be positive.")
            if self.warmup_patience <= 0:
                raise ValueError("warmup_patience must be positive.")
            if self.warmup_max_steps <= 0:
                raise ValueError("warmup_max_steps must be positive.")
            if self.warmup_mc_eps <= 0:
                raise ValueError("warmup_mc_eps must be positive.")
            if self.warmup_mc_min_delta < 0:
                raise ValueError("warmup_mc_min_delta must be non-negative.")

        # --------------------------------------------------------------
        # Teacher and adaptive-threshold state.
        # --------------------------------------------------------------
        self._warmup_teacher_state = None
        self._current_teacher_margin = None

        # Minimum margin observed during warm-up; later used to initialize
        # tau_base. In automatic mode we snapshot it together with the best
        # warm-up teacher candidate so the two correspond to the same point.
        self._warmup_min_margin = None
        self._tau_base = None
        self._tau_prev = None

        self._current_adaptive_threshold = None
        self._current_D = None
        self._current_alpha = None
        self._current_candidate_ratio = None

        # --------------------------------------------------------------
        # Automatic warm-up state machine.
        # --------------------------------------------------------------
        self._warmup_finished = False
        self._probe_dataset = None
        self._probe_dataloader = None
        self._previous_probe_margins = None
        self._mc_history = deque(maxlen=self.warmup_ma_window)
        self._best_mc_bar = float("inf")
        self._warmup_bad_count = 0
        self._last_probe_eval_step = None
        self._last_probe_step = None
        self._best_warmup_step = None
        self._warmup_stop_step = None
        self._warmup_stop_reason = None
        self._best_warmup_teacher_state = None
        self._best_warmup_min_margin = None
        self._current_mc = None
        self._current_mc_bar = None

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

            # Fixed-step mode remains available as an ablation/fallback.
            if not self.enable_auto_warmup and self.margin_warmup_steps <= 0:
                raise ValueError(
                    "Fixed warm-up requires margin_warmup_steps > 0. "
                    "Use --enable_auto_warmup to determine the stopping point "
                    "from margin changes automatically."
                )

        if self.enable_auto_warmup:
            self._init_auto_warmup_probe()

    def _get_beta(self):
        beta = getattr(self, "beta", None)
        if beta is None:
            beta = getattr(self.args, "beta", 1.0)
        if isinstance(beta, (list, tuple)):
            beta = beta[0]
        return float(beta)

    def _get_trainable_params(self, model):
        params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]

        if len(params) == 0:
            params = list(model.named_parameters())

        return params

    def _is_main_process(self):
        if hasattr(self, "accelerator") and self.accelerator is not None:
            return self.accelerator.is_main_process
        return True

    def _init_auto_warmup_probe(self):
        """
        Build a fixed probe set D_m from the already preprocessed training
        dataset and remove it from the actual training dataset.

        Doing the split after DPOTrainer.__init__ is intentional: at this point
        the dataset has already passed through the same DPO preprocessing as
        the training data, so the existing data collator can be reused safely.
        """
        dataset = getattr(self, "train_dataset", None)
        if dataset is None:
            raise RuntimeError("Automatic warm-up requires a training dataset.")
        if not hasattr(dataset, "select"):
            raise TypeError(
                "Automatic warm-up currently requires a Hugging Face Dataset "
                "with a select() method."
            )

        dataset_size = len(dataset)
        if dataset_size < 2:
            raise ValueError(
                "The training dataset must contain at least two samples so that "
                "a non-empty probe set and a non-empty training set can be built."
            )

        probe_size = min(self.warmup_probe_size, dataset_size - 1)
        if probe_size != self.warmup_probe_size and self._is_main_process():
            print(
                f"[AutoWarmup] Requested probe_size={self.warmup_probe_size}, "
                f"but dataset_size={dataset_size}; using probe_size={probe_size}."
            )

        generator = torch.Generator()
        generator.manual_seed(self.warmup_probe_seed)
        permutation = torch.randperm(dataset_size, generator=generator).tolist()

        # Sort the selected indices to make the probe order fixed and stable.
        probe_indices = sorted(permutation[:probe_size])
        train_indices = sorted(permutation[probe_size:])

        self._probe_dataset = dataset.select(probe_indices)
        self.train_dataset = dataset.select(train_indices)

        probe_batch_size = self.warmup_probe_batch_size
        if probe_batch_size == 0:
            probe_batch_size = int(
                getattr(self.args, "per_device_eval_batch_size", 1)
            )
        probe_batch_size = max(1, probe_batch_size)

        # Every rank evaluates the same full probe set in the same order.
        # This deliberately avoids a DistributedSampler because the margin
        # vector M_t must keep an identical sample ordering across probe checks.
        self._probe_dataloader = DataLoader(
            self._probe_dataset,
            batch_size=probe_batch_size,
            sampler=SequentialSampler(self._probe_dataset),
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )

        if self._is_main_process():
            print("=" * 80)
            print("[AutoWarmup] Fixed probe set initialized.")
            print(f"[AutoWarmup] probe_size={len(self._probe_dataset)}")
            print(f"[AutoWarmup] remaining_train_size={len(self.train_dataset)}")
            print(f"[AutoWarmup] probe_batch_size={probe_batch_size}")
            print(f"[AutoWarmup] probe_seed={self.warmup_probe_seed}")
            print("=" * 80)

    def _clone_trainable_param_state(self, model):
        return {
            name: param.detach().clone()
            for name, param in self._get_trainable_params(model)
        }

    def _is_warmup_finished(self):
        """
        Unified warm-up gate.

        Automatic mode uses the MC-based state machine. Fixed-step mode keeps
        the original behavior for ablations and backward compatibility.
        """
        if not self.enable_margin_threshold:
            return True

        if self.enable_auto_warmup:
            return self._warmup_finished

        current_step = int(getattr(self.state, "global_step", 0))
        return current_step >= self.margin_warmup_steps

    def _compute_probe_margins(self, model):
        """
        Compute the DPO preference-margin vector M_t on the fixed probe set D_m:

            M_i^t = beta * [(log pi_theta(y_w|x) - log pi_theta(y_l|x))
                            - (log pi_ref(y_w|x) - log pi_ref(y_l|x))].

        The model is temporarily switched to eval mode so dropout does not add
        artificial variation to MC_t. The original train/eval mode is restored.
        """
        if self._probe_dataloader is None:
            raise RuntimeError("The automatic warm-up probe dataloader is None.")

        all_margins = []
        was_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                for probe_batch in self._probe_dataloader:
                    probe_batch = self._prepare_inputs(probe_batch)

                    policy_output = self.concatenated_forward(model, probe_batch)
                    policy_chosen_logps, policy_rejected_logps = (
                        self._unpack_concatenated_forward_output(policy_output)
                    )

                    if getattr(self, "ref_model", None) is None:
                        with self.null_ref_context():
                            ref_output = self.concatenated_forward(model, probe_batch)
                    else:
                        ref_output = self.concatenated_forward(
                            self.ref_model,
                            probe_batch,
                        )

                    reference_chosen_logps, reference_rejected_logps = (
                        self._unpack_concatenated_forward_output(ref_output)
                    )

                    pi_logratios = policy_chosen_logps - policy_rejected_logps
                    ref_logratios = (
                        reference_chosen_logps - reference_rejected_logps
                    )
                    margins = self._get_beta() * (pi_logratios - ref_logratios)
                    all_margins.append(margins.detach().float().cpu())
        finally:
            if was_training:
                model.train()
            else:
                model.eval()

        if not all_margins:
            raise RuntimeError("No margins were produced for the probe set.")

        return torch.cat(all_margins, dim=0)

    def _compute_margin_change(self, current_margins, previous_margins):
        """
        Numerically stable version of Eq. (10):

            MC_t = mean_i |M_i^t - M_i^{t-1}| / (|M_i^{t-1}| + eps).

        The absolute denominator and eps preserve the intended relative-change
        interpretation while avoiding explosions when a previous margin is near
        zero or negative.
        """
        if current_margins.shape != previous_margins.shape:
            raise RuntimeError(
                "Probe margin shape changed between checks: "
                f"current={current_margins.shape}, "
                f"previous={previous_margins.shape}."
            )

        relative_change = (
            (current_margins - previous_margins).abs()
            / (previous_margins.abs() + self.warmup_mc_eps)
        )
        return relative_change.mean()

    def _save_best_warmup_candidate(self, model, current_step, mc_bar):
        """Save theta_warm and the corresponding tau_base statistics."""
        self._best_mc_bar = float(mc_bar)
        self._warmup_bad_count = 0
        self._best_warmup_step = int(current_step)
        self._best_warmup_teacher_state = self._clone_trainable_param_state(model)

        if self._warmup_min_margin is not None:
            self._best_warmup_min_margin = (
                self._warmup_min_margin.detach().clone()
            )

        if self._is_main_process():
            print(
                f"[AutoWarmup] New best MC_bar={self._best_mc_bar:.8f} "
                f"at step={self._best_warmup_step}; "
                "saved teacher candidate."
            )

    def _finish_auto_warmup(self, model, current_step, reason):
        """
        Finalize automatic warm-up.

        The best historical model is used as the frozen teacher, while the
        current student is NOT rolled back. This avoids an inconsistency between
        restored model parameters and future Adam/AdamW optimizer states.
        """
        if self._warmup_finished:
            return

        # A max-step fallback may trigger before a full moving-average window.
        # In that case, use the current model as a valid teacher candidate.
        if self._best_warmup_teacher_state is None:
            self._best_warmup_step = int(current_step)
            self._best_warmup_teacher_state = self._clone_trainable_param_state(model)
            if self._warmup_min_margin is not None:
                self._best_warmup_min_margin = (
                    self._warmup_min_margin.detach().clone()
                )

        self._warmup_teacher_state = {
            name: tensor.detach().clone()
            for name, tensor in self._best_warmup_teacher_state.items()
        }

        if self._best_warmup_min_margin is not None:
            self._warmup_min_margin = (
                self._best_warmup_min_margin.detach().clone()
            )

        # Force tau_base/tau_prev to be initialized from the statistics that
        # correspond to the selected best warm-up teacher candidate.
        self._tau_base = None
        self._tau_prev = None
        self._finalize_tau_base()

        self._warmup_finished = True
        self._warmup_stop_step = int(current_step)
        self._warmup_stop_reason = str(reason)

        if self._is_main_process():
            print("=" * 80)
            print("[AutoWarmup] Warm-up finished.")
            print(f"[AutoWarmup] stop_step={self._warmup_stop_step}")
            print(f"[AutoWarmup] stop_reason={self._warmup_stop_reason}")
            print(f"[AutoWarmup] best_step={self._best_warmup_step}")
            print(f"[AutoWarmup] best_MC_bar={self._best_mc_bar}")
            print(
                "[AutoWarmup] The best historical checkpoint is frozen as "
                "teacher; the current student continues without rollback."
            )
            print("=" * 80)

    def _maybe_update_auto_warmup(self, model):
        """
        Implement Algorithm 1, Lines 3-12:

          M_0 -> train for n steps -> M_t -> MC_t -> moving average MC_bar_t
              -> save a new best teacher candidate when MC_bar_t decreases
              -> stop after p consecutive non-improving probe checks.
        """
        if not self.enable_auto_warmup or self._warmup_finished:
            return

        current_step = int(getattr(self.state, "global_step", 0))

        # Gradient accumulation may call get_batch_loss_metrics multiple times
        # at the same optimizer step. Probe at most once per global_step.
        if self._last_probe_eval_step == current_step:
            return

        reached_max_steps = current_step >= self.warmup_max_steps
        should_probe = (
            current_step == 0
            or current_step % self.warmup_probe_interval == 0
            or reached_max_steps
        )
        if not should_probe:
            return

        current_margins = self._compute_probe_margins(model)
        self._last_probe_eval_step = current_step
        self._last_probe_step = current_step

        # Algorithm initialization: compute M_0 once before any update.
        if self._previous_probe_margins is None:
            self._previous_probe_margins = current_margins
            if self._is_main_process():
                print(
                    f"[AutoWarmup] Initialized M_0 on D_m at step={current_step}."
                )
            return

        mc_t_tensor = self._compute_margin_change(
            current_margins=current_margins,
            previous_margins=self._previous_probe_margins,
        )
        self._previous_probe_margins = current_margins

        mc_t = float(mc_t_tensor.item())
        self._current_mc = mc_t
        self._mc_history.append(mc_t)

        if len(self._mc_history) == self.warmup_ma_window:
            mc_bar = sum(self._mc_history) / len(self._mc_history)
            self._current_mc_bar = float(mc_bar)

            improved = (
                mc_bar
                < self._best_mc_bar - self.warmup_mc_min_delta
            )

            if improved:
                self._save_best_warmup_candidate(
                    model=model,
                    current_step=current_step,
                    mc_bar=mc_bar,
                )
            else:
                self._warmup_bad_count += 1

            if self._is_main_process():
                print(
                    f"[AutoWarmup] step={current_step}, MC_t={mc_t:.8f}, "
                    f"MC_bar={mc_bar:.8f}, best={self._best_mc_bar:.8f}, "
                    f"bad_count={self._warmup_bad_count}/"
                    f"{self.warmup_patience}"
                )

            if self._warmup_bad_count >= self.warmup_patience:
                self._finish_auto_warmup(
                    model=model,
                    current_step=current_step,
                    reason="patience",
                )
                return
        elif self._is_main_process():
            print(
                f"[AutoWarmup] step={current_step}, MC_t={mc_t:.8f}, "
                f"collecting moving-average window "
                f"({len(self._mc_history)}/{self.warmup_ma_window})."
            )

        if reached_max_steps and not self._warmup_finished:
            self._finish_auto_warmup(
                model=model,
                current_step=current_step,
                reason="max_steps",
            )

    def _compute_gradient_norm_penalty(self, per_sample_losses, model):
        """
        Exact per-sample gradient-norm penalty matching:

            mean_i || grad_theta L_i ||_2^2

        where theta contains only trainable parameters. Therefore, when LoRA is
        enabled, the penalty is computed over LoRA parameters only.

        Important: create_graph=True is required. Otherwise the penalty is
        detached from theta and will not affect parameter updates.
        """
        trainable_params = [
            param
            for _, param in self._get_trainable_params(model)
            if param.requires_grad
        ]

        if len(trainable_params) == 0:
            raise RuntimeError(
                "No trainable parameters found for gradient regularization."
            )

        flat_losses = per_sample_losses.reshape(-1)
        penalties = []

        for loss_i in flat_losses:
            grads = torch.autograd.grad(
                outputs=loss_i,
                inputs=trainable_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

            penalty_i = None
            for grad in grads:
                if grad is None:
                    continue

                # Accumulate in fp32 for numerical stability while preserving
                # the higher-order autograd graph.
                grad_sq_sum = grad.float().pow(2).sum()
                penalty_i = (
                    grad_sq_sum
                    if penalty_i is None
                    else penalty_i + grad_sq_sum
                )

            if penalty_i is None:
                raise RuntimeError(
                    "All gradients are None while computing gradient penalty."
                )

            penalties.append(penalty_i)

        return torch.stack(penalties).mean()

    def _maybe_all_reduce_min(self, value):
        """
        多卡训练时同步 warmup_min_margin，保证所有 rank 的 tau_base 一致。
        """
        if not torch.is_tensor(value):
            return value

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            value = value.clone()
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN)

        return value

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

    def _update_warmup_min_margin(self, margin):
        """
        warmup 阶段记录全局最小 margin。
        """
        with torch.no_grad():
            local_min = margin.detach().float().min()
            global_min = self._maybe_all_reduce_min(local_min)

            if self._warmup_min_margin is None:
                self._warmup_min_margin = global_min.detach().clone()
            else:
                self._warmup_min_margin = torch.minimum(
                    self._warmup_min_margin.to(global_min.device),
                    global_min,
                ).detach()

    def _finalize_tau_base(self):
        """
        warmup 结束后，将 warmup_min_margin 固定为 tau_base。
        """
        if self._tau_base is not None:
            return

        if self._warmup_min_margin is None:
            raise RuntimeError(
                "Cannot initialize tau_base because _warmup_min_margin is None. "
                "Please check whether warmup stage has run correctly."
            )

        self._tau_base = self._warmup_min_margin.detach().clone()
        self._tau_prev = self._tau_base.detach().clone()

        if self._is_main_process():
            print("=" * 80)
            print("[MarginThresholdDPOTrainer] Adaptive threshold initialized.")
            print(f"tau_base = warmup_min_margin = {self._tau_base.item():.8f}")
            print(f"tau_prev = {self._tau_prev.item():.8f}")
            print("=" * 80)

    def _maybe_capture_warmup_teacher(self, model):
        if self._warmup_teacher_state is not None:
            return

        self._finalize_tau_base()

        if self._is_main_process():
            print("=" * 80)
            print(
                f"[MarginThresholdDPOTrainer] Capturing warmup teacher at "
                f"global_step={getattr(self.state, 'global_step', 0)}"
            )
            print("=" * 80)

        params = self._get_trainable_params(model)

        self._warmup_teacher_state = {
            name: param.detach().clone()
            for name, param in params
        }

        if self._is_main_process():
            print(
                f"[MarginThresholdDPOTrainer] Captured "
                f"{len(self._warmup_teacher_state)} teacher tensors."
            )
            print("=" * 80)

    def _load_param_state_(self, model, state_dict):
        named_params = dict(model.named_parameters())

        for name, tensor in state_dict.items():
            if name not in named_params:
                raise KeyError(f"Parameter {name} not found in current model.")

            param = named_params[name]

            if param.shape != tensor.shape:
                raise RuntimeError(
                    f"Shape mismatch for parameter {name}: "
                    f"model={param.shape}, state={tensor.shape}"
                )

            param.data.copy_(tensor.data.to(param.device))

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

    def _forward_with_warmup_teacher(self, model, batch):
        if self._warmup_teacher_state is None:
            raise RuntimeError("Warmup teacher has not been captured yet.")

        params = self._get_trainable_params(model)

        current_state = {
            name: param.detach().clone()
            for name, param in params
        }

        was_training = model.training

        try:
            self._load_param_state_(model, self._warmup_teacher_state)

            model.eval()
            with torch.no_grad():
                teacher_output = self.concatenated_forward(model, batch)

        finally:
            self._load_param_state_(model, current_state)

            if was_training:
                model.train()
            else:
                model.eval()

        return teacher_output

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

    # def _compute_adaptive_threshold(
    #     self,
    #     batch,
    #     teacher_margin,
    #     reference_chosen_logps,
    #     reference_rejected_logps,
    # ):
    #     """
    #     关键实现:
    #
    #     D_t(B) 中的筛选使用更新前阈值 tau_prev:
    #
    #         D_t(B) =
    #             sum_i I(m_i < tau_prev) *
    #             | log pi_ref(y_w|x) / |y_w|
    #              - log pi_ref(y_l|x) / |y_l| |
    #
    #     然后:
    #
    #         alpha_t(B) = 1 - sigmoid(D_t(B))
    #         tau_t(B) = alpha_t(B) * tau_base
    #
    #     其中:
    #
    #         tau_base = warmup 阶段观测到的最小 margin
    #         tau_prev = 上一个 batch 更新后的 tau
    #     """
    #
    #     if self._tau_base is None or self._tau_prev is None:
    #         raise RuntimeError(
    #             "tau_base or tau_prev is None. "
    #             "Please make sure warmup has finished and tau has been initialized."
    #         )
    #
    #     with torch.no_grad():
    #         tau_prev = self._tau_prev.detach().to(
    #             device=teacher_margin.device,
    #             dtype=teacher_margin.dtype,
    #         )
    #
    #         tau_base = self._tau_base.detach().to(
    #             device=teacher_margin.device,
    #             dtype=teacher_margin.dtype,
    #         )
    #
    #         chosen_len, rejected_len = self._get_response_lengths(
    #             batch,
    #             reference_chosen_logps,
    #         )
    #
    #         ref_chosen_avg_logp = reference_chosen_logps / chosen_len
    #         ref_rejected_avg_logp = reference_rejected_logps / rejected_len
    #
    #         distinguishability = torch.abs(
    #             ref_chosen_avg_logp - ref_rejected_avg_logp
    #         )
    #
    #         # 这里必须用“更新前的 tau_prev”筛选，而不是新 tau，也不是人为固定 tau。
    #         candidate_mask = teacher_margin < tau_prev
    #
    #         local_D = (
    #             candidate_mask.float() * distinguishability
    #         ).sum()
    #
    #         # 多卡同步 D_t(B)，让所有 rank 使用同一个 tau_t。
    #         D_t = self._maybe_all_reduce_sum(local_D)
    #
    #         alpha_t = 1.0 - torch.sigmoid(D_t)
    #
    #         tau_t = alpha_t * tau_base
    #
    #         candidate_ratio = candidate_mask.float().mean()
    #
    #     return (
    #         tau_t.detach(),
    #         D_t.detach(),
    #         alpha_t.detach(),
    #         candidate_ratio.detach(),
    #         tau_prev.detach(),
    #         tau_base.detach(),
    #     )

    def _compute_adaptive_threshold(
            self,
            batch,
            teacher_margin,
            reference_chosen_logps,
            reference_rejected_logps,
    ):
        """
        关键实现:

        D_t(B) 中的筛选使用更新前阈值 tau_prev:

            D_t(B) =
                sum_i I(m_i < tau_prev) *
                | log pi_ref(y_w|x) / |y_w|
                 - log pi_ref(y_l|x) / |y_l| |

        然后:

            如果没有筛选到候选噪声样本:
                alpha_t(B) = 0

            否则:
                alpha_t(B) = 1 - sigmoid(D_t(B))

            tau_t(B) = alpha_t(B) * tau_base

        其中:

            tau_base = warmup 阶段观测到的最小 margin
            tau_prev = 上一个 batch 更新后的 tau
        """

        if self._tau_base is None or self._tau_prev is None:
            raise RuntimeError(
                "tau_base or tau_prev is None. "
                "Please make sure warmup has finished and tau has been initialized."
            )

        with torch.no_grad():
            tau_prev = self._tau_prev.detach().to(
                device=teacher_margin.device,
                dtype=teacher_margin.dtype,
            )

            tau_base = self._tau_base.detach().to(
                device=teacher_margin.device,
                dtype=teacher_margin.dtype,
            )

            chosen_len, rejected_len = self._get_response_lengths(
                batch,
                reference_chosen_logps,
            )

            ref_chosen_avg_logp = reference_chosen_logps / chosen_len
            ref_rejected_avg_logp = reference_rejected_logps / rejected_len

            distinguishability = torch.abs(
                ref_chosen_avg_logp - ref_rejected_avg_logp
            )

            # 用“更新前的 tau_prev”筛选候选噪声样本
            candidate_mask = teacher_margin < tau_prev

            local_D = (
                    candidate_mask.float() * distinguishability
            ).sum()

            # 新增：统计当前 rank 上筛选到的候选噪声样本数量
            local_candidate_count = candidate_mask.float().sum()

            # 新增：统计当前 rank 上的样本总数，用于计算全局 candidate_ratio
            local_total_count = torch.tensor(
                candidate_mask.numel(),
                device=teacher_margin.device,
                dtype=teacher_margin.dtype,
            )

            # 多卡同步 D_t(B)，让所有 rank 使用同一个 tau_t
            D_t = self._maybe_all_reduce_sum(local_D)

            # 新增：多卡同步候选噪声样本数量
            candidate_count = self._maybe_all_reduce_sum(local_candidate_count)
            total_count = self._maybe_all_reduce_sum(local_total_count)

            # 关键修改：
            # 如果没有筛选到候选噪声样本，则 alpha_t 直接等于 0。
            # 否则按照原公式 alpha_t = 1 - sigmoid(D_t)。
            if candidate_count.item() <= 0:
                alpha_t = torch.zeros_like(D_t)
            else:
                alpha_t = 1.0 - torch.sigmoid(D_t)

            tau_t = alpha_t * tau_base

            # 建议使用全局 candidate_ratio，而不是当前 rank 的局部 mean
            candidate_ratio = candidate_count / total_count.clamp_min(1.0)

        return (
            tau_t.detach(),
            D_t.detach(),
            alpha_t.detach(),
            candidate_ratio.detach(),
            tau_prev.detach(),
            tau_base.detach(),
        )

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

        warmup_finished = self._is_warmup_finished()

        if (
            not self.enable_margin_threshold
            or not warmup_finished
            or not self.model.training
        ):
            if (
                self.enable_margin_threshold
                and self.model.training
                and not warmup_finished
            ):
                self._update_warmup_min_margin(student_margin)

            return normal_losses, chosen_rewards, rejected_rewards

        if self._current_teacher_margin is None:
            raise RuntimeError(
                "Teacher margin is None. "
                "get_batch_loss_metrics must compute teacher margin before dpo_loss."
            )

        if self._current_adaptive_threshold is None:
            raise RuntimeError(
                "Adaptive threshold is None. "
                "get_batch_loss_metrics must compute adaptive threshold before dpo_loss."
            )

        with torch.no_grad():
            teacher_margin = self._current_teacher_margin.detach().to(normal_losses.device)
            adaptive_threshold = self._current_adaptive_threshold.detach().to(normal_losses.device)

            if teacher_margin.shape != normal_losses.shape:
                raise RuntimeError(
                    f"teacher_margin shape {teacher_margin.shape} does not match "
                    f"normal_losses shape {normal_losses.shape}"
                )

            noisy_mask = teacher_margin <= adaptive_threshold

        losses = torch.where(noisy_mask, flipped_losses, normal_losses)

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        current_step = int(getattr(self.state, "global_step", 0))

        # Automatic warm-up decisions are made before the current training-batch
        # forward pass. Probe evaluation runs under no_grad/eval mode and then
        # restores the model to training mode, so the subsequent graph is clean.
        if (
            train_eval == "train"
            and model.training
            and self.enable_margin_threshold
            and self.enable_auto_warmup
        ):
            self._maybe_update_auto_warmup(model)

        warmup_finished = self._is_warmup_finished()

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

        self._current_teacher_margin = None
        self._current_adaptive_threshold = None
        self._current_D = None
        self._current_alpha = None
        self._current_candidate_ratio = None
        self._current_tau_prev_for_log = None
        self._current_tau_base_for_log = None

        if (
            self.enable_margin_threshold
            and self.model.training
            and warmup_finished
        ):
            self._maybe_capture_warmup_teacher(model)

            teacher_output = self._forward_with_warmup_teacher(model, batch)
            teacher_chosen_logps, teacher_rejected_logps = (
                self._unpack_concatenated_forward_output(teacher_output)
            )

            beta = self._get_beta()

            with torch.no_grad():
                teacher_pi_logratios = teacher_chosen_logps - teacher_rejected_logps
                ref_logratios = reference_chosen_logps - reference_rejected_logps
                teacher_logits = teacher_pi_logratios - ref_logratios
                self._current_teacher_margin = beta * teacher_logits

                (
                    tau_t,
                    D_t,
                    alpha_t,
                    candidate_ratio,
                    tau_prev,
                    tau_base,
                ) = self._compute_adaptive_threshold(
                    batch=batch,
                    teacher_margin=self._current_teacher_margin,
                    reference_chosen_logps=reference_chosen_logps,
                    reference_rejected_logps=reference_rejected_logps,
                )

                self._current_adaptive_threshold = tau_t
                self._current_D = D_t
                self._current_alpha = alpha_t
                self._current_candidate_ratio = candidate_ratio
                self._current_tau_prev_for_log = tau_prev
                self._current_tau_base_for_log = tau_base

                # 更新 tau_prev，供下一次 batch 计算 D 使用。
                self._tau_prev = tau_t.detach().clone()

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        # The per-sample `losses` returned by dpo_loss are exactly the
        # MNC-DPO losses after warm-up: clean pairs use normal DPO and
        # selected noisy pairs use the flipped objective.
        mnc_dpo_loss_mean = losses.mean()
        total_loss = mnc_dpo_loss_mean

        grad_norm_penalty = None
        grad_reg_weight = None

        # Eq. (11):
        #   L_MNCS-DPO = E[ l_MNC-DPO
        #                    + (alpha_t / lambda) * ||grad_theta l_MNC-DPO||_2^2 ]
        #
        # alpha_t only exists after warm-up, so the regularizer is activated
        # only in the correction stage. During warm-up, training remains
        # standard DPO.
        if (
            self.enable_grad_regularization
            and train_eval == "train"
            and model.training
            and torch.is_grad_enabled()
            and warmup_finished
            and self._current_alpha is not None
        ):
            grad_norm_penalty = self._compute_gradient_norm_penalty(
                per_sample_losses=losses,
                model=model,
            )

            # alpha_t is a batch/step-level adaptive scalar produced by the
            # noise-selection module. It is detached from theta by design.
            alpha_t = self._current_alpha.detach().to(
                device=grad_norm_penalty.device,
                dtype=grad_norm_penalty.dtype,
            )
            grad_reg_weight = alpha_t / self.grad_reg_lambda

            total_loss = (
                mnc_dpo_loss_mean
                + grad_reg_weight * grad_norm_penalty
            )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margins = chosen_rewards - rejected_rewards

        prefix = "eval_" if train_eval == "eval" else ""

        metrics = {}

        if self.enable_auto_warmup and train_eval == "train":
            metrics["warmup/finished"] = torch.tensor(
                float(self._warmup_finished)
            )
            metrics["warmup/bad_count"] = torch.tensor(
                float(self._warmup_bad_count)
            )
            metrics["warmup/probe_size"] = torch.tensor(
                float(len(self._probe_dataset))
            )
            if self._last_probe_step is not None:
                metrics["warmup/last_probe_step"] = torch.tensor(
                    float(self._last_probe_step)
                )
            if self._current_mc is not None:
                metrics["warmup/mc"] = torch.tensor(float(self._current_mc))
            if self._current_mc_bar is not None:
                metrics["warmup/mc_bar"] = torch.tensor(
                    float(self._current_mc_bar)
                )
            if self._best_mc_bar != float("inf"):
                metrics["warmup/best_mc_bar"] = torch.tensor(
                    float(self._best_mc_bar)
                )
            if self._best_warmup_step is not None:
                metrics["warmup/best_step"] = torch.tensor(
                    float(self._best_warmup_step)
                )
            if self._warmup_stop_step is not None:
                metrics["warmup/stop_step"] = torch.tensor(
                    float(self._warmup_stop_step)
                )

        metrics[f"{prefix}loss/mnc_dpo"] = (
            mnc_dpo_loss_mean.detach().float().cpu()
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
            and self._current_teacher_margin is not None
            and self._current_adaptive_threshold is not None
        ):
            with torch.no_grad():
                teacher_margin = self._current_teacher_margin.detach()
                final_noisy_ratio = (
                    teacher_margin <= self._current_adaptive_threshold.to(teacher_margin.device)
                ).float().mean()

            metrics[f"{prefix}teacher_margin/mean"] = teacher_margin.mean().detach().cpu()
            metrics[f"{prefix}teacher_margin/min"] = teacher_margin.min().detach().cpu()
            metrics[f"{prefix}teacher_margin/max"] = teacher_margin.max().detach().cpu()

            metrics[f"{prefix}adaptive_threshold/tau_base"] = self._current_tau_base_for_log.detach().cpu()
            metrics[f"{prefix}adaptive_threshold/tau_prev"] = self._current_tau_prev_for_log.detach().cpu()
            metrics[f"{prefix}adaptive_threshold/tau_t"] = self._current_adaptive_threshold.detach().cpu()
            metrics[f"{prefix}adaptive_threshold/alpha"] = self._current_alpha.detach().cpu()
            metrics[f"{prefix}adaptive_threshold/D"] = self._current_D.detach().cpu()
            metrics[f"{prefix}adaptive_threshold/candidate_ratio"] = self._current_candidate_ratio.detach().cpu()
            metrics[f"{prefix}adaptive_threshold/final_noisy_ratio"] = final_noisy_ratio.detach().cpu()

        return total_loss, metrics

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
    "--enable_auto_warmup",
    action="store_true",
    help=(
        "Automatically stop warm-up using moving-averaged margin changes on "
        "a fixed probe set."
    ),
)
parser.add_argument(
    "--warmup_probe_size",
    type=int,
    default=256,
)
parser.add_argument(
    "--warmup_probe_batch_size",
    type=int,
    default=0,
    help="0 means use per_device_eval_batch_size.",
)
parser.add_argument(
    "--warmup_probe_interval",
    type=int,
    default=200,
    help="n: optimizer steps between two probe-margin evaluations.",
)
parser.add_argument(
    "--warmup_ma_window",
    type=int,
    default=5,
    help="k: moving-average window over MC_t.",
)
parser.add_argument(
    "--warmup_patience",
    type=int,
    default=3,
    help="p: consecutive non-improving MC_bar checks before stopping warm-up.",
)
parser.add_argument(
    "--warmup_max_steps",
    type=int,
    default=5000,
    help="Safety fallback that prevents warm-up from running indefinitely.",
)
parser.add_argument(
    "--warmup_mc_eps",
    type=float,
    default=1e-6,
    help="Numerical epsilon in the relative margin-change denominator.",
)
parser.add_argument(
    "--warmup_mc_min_delta",
    type=float,
    default=0.0,
    help="Minimum MC_bar decrease required to reset patience.",
)
parser.add_argument(
    "--warmup_probe_seed",
    type=int,
    default=42,
)
parser.add_argument(
    "--enable_grad_regularization",
    action="store_true",
)
parser.add_argument(
    "--grad_reg_lambda",
    type=float,
    default=10.0,
    help=(
        "Positive lambda in alpha_t / lambda. With the current "
        "alpha_t = 1 - sigmoid(D_t) and D_t >= 0, alpha_t <= 0.5; "
        "lambda=10 therefore gives a maximum regularization weight of 0.05."
    ),
)

args = parser.parse_args_into_dataclasses()
model_args, data_args, pipeline_args = args[:3]
method_args = parser.parse_args()

print("&&&&&&&&&&&&&&&&&&&")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    load_in_8bit=True,
)
tokenizer.pad_token = tokenizer.eos_token


print("Loading model...")
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
    or method_args.enable_auto_warmup
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
    bf16=pipeline_args.bf16,
    seed=pipeline_args.seed,
    do_train=pipeline_args.do_train,
    do_eval=pipeline_args.do_eval,
    run_name=pipeline_args.run_name,
    eval_strategy=pipeline_args.eval_strategy,
    eval_steps=pipeline_args.eval_steps,
    save_steps=pipeline_args.save_steps,
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
            enable_auto_warmup=method_args.enable_auto_warmup,
            warmup_probe_size=method_args.warmup_probe_size,
            warmup_probe_batch_size=method_args.warmup_probe_batch_size,
            warmup_probe_interval=method_args.warmup_probe_interval,
            warmup_ma_window=method_args.warmup_ma_window,
            warmup_patience=method_args.warmup_patience,
            warmup_max_steps=method_args.warmup_max_steps,
            warmup_mc_eps=method_args.warmup_mc_eps,
            warmup_mc_min_delta=method_args.warmup_mc_min_delta,
            warmup_probe_seed=method_args.warmup_probe_seed,
        )
    )


trainer = trainer_class(**trainer_kwargs)

trainer.train()

if model_args.use_lora:
    model_lora = model_lora.merge_and_unload()

model_lora.save_pretrained(
    pipeline_args.output_dir,
    safe_serialization=False,
)
model.config.save_pretrained(pipeline_args.output_dir)
tokenizer.save_pretrained(pipeline_args.output_dir)