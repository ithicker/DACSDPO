#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode=${1:?Usage: run_dacsdpo.sh dpo|dac|dpo_regularizer|dacs SEED [PATIENCE]}
seed=${2:?Specify training seed}
patience=${3:-50}
model_name_or_path=models/UltraFeedback/sft_DPO_pythia2_no_clean
dataset_path=datasets/UltraFeedback/no_clean_0.4
output_dir="models/UltraFeedback/${mode}_release_uf40_seed${seed}"
batch_size=4
max_steps=5140
beta=0.01
tau=-0.5
grad_reg_lambda=10
fd_pairs=1
fd_radius=0.05

extra=()
case "$mode" in
  dpo) ;;
  dac) extra+=(--enable_margin_threshold) ;;
  dpo_regularizer) extra+=(--enable_margin_threshold --enable_grad_regularization --disable_correction) ;;
  dacs) extra+=(--enable_margin_threshold --enable_grad_regularization) ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

mkdir -p "$output_dir"
export WANDB_MODE=disabled WANDB_DISABLED=true OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
started=$SECONDS
trap 'code=$?; printf "exit_code=%s\njob_wall_seconds=%s\n" "$code" "$((SECONDS-started))" > "$output_dir/process_status.txt"' EXIT

common=(
  --opt_alg DPO
  --model_name_or_path "$model_name_or_path"
  --dataset_path "$dataset_path"
  --output_dir "$output_dir"
  --num_train_epochs 3 --max_steps "$max_steps" --learning_rate 5e-5
  --block_size 512 --per_device_train_batch_size "$batch_size"
  --per_device_eval_batch_size "$batch_size"
  --deepspeed configs/ds_config_zero2.json --bf16
  --validation_split_percentage 0 --logging_steps 10 --do_train
  --use_lora 1 --lora_r 16 --lr_scheduler_type constant
  --save_steps 999999 --eval_strategy steps --eval_steps 999999
  --weight_decay 0.0001 --dataloader_num_workers 1
  --seed "$seed" --beta "$beta" --loss_type sigmoid
)

if [[ "$mode" == dpo ]]; then
  argv=(python -m torch.distributed.run --standalone --nproc_per_node=8
        src/lm_flow/train_dacsdpo.py "${common[@]}")
else
  # The core Python file runs patience warm-up and resumes the best checkpoint.
  argv=(python src/lm_flow/train_dacsdpo.py --auto_warmup "${common[@]}"
        --margin_threshold "$tau" --warmup_patience "$patience"
        --warmup_probe_size 20 --warmup_window 10 --warmup_epsilon 1e-6
        --grad_reg_lambda "$grad_reg_lambda" --fd_pairs "$fd_pairs"
        --fd_radius "$fd_radius" "${extra[@]}")
fi

printf '%q ' "${argv[@]}" > "$output_dir/launch_argv.txt"
printf '\n' >> "$output_dir/launch_argv.txt"
"${argv[@]}" 2>&1 | tee "$output_dir/console.log"
