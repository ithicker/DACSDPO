#!/bin/bash

set -e

deepspeed_args="--master_port=17903 --include=localhost:0,1"

opt_alg=${1:-DPO}
dataset=${2:-AnthropicHH}
version=${3:-no_clean}
pretty_name=${4:-DPO_llama3_8b_no_clean}
model_path="${5:-meta-llama/Meta-Llama-3-8B}"

data_path=datasets/${dataset}/${version}
model_name_or_path=models/${dataset}/sft_${pretty_name}
output_dir=models/${dataset}/${pretty_name}

bsz=2
num_train_epochs=1.0
loss_type="sigmoid"
beta=0.1

# ==========================================================
# Automatic warm-up (Algorithm 1, Lines 3-12)
# ==========================================================
# A fixed probe subset D_m is removed from the training set inside the trainer.
warmup_probe_size=256
warmup_probe_batch_size=${bsz}

# n: evaluate probe margins every n optimizer steps.
warmup_probe_interval=100

# k: moving-average window for MC_t.
warmup_ma_window=10

# p: stop after p consecutive non-improving moving-average checks.
warmup_patience=5

# Safety fallback in case no local minimum is detected early enough.
warmup_max_steps=4000

# Numerical stabilization and optional improvement tolerance.
warmup_mc_eps=1e-6
warmup_mc_min_delta=0.0
warmup_probe_seed=14

# Fixed-step warm-up remains available in the Python code as an ablation/fallback.
# It is ignored when --enable_auto_warmup is enabled.
margin_warmup_steps=2000

# ==========================================================
# MNCS-DPO stability regularization
# L = L_MNC-DPO + (alpha_t / lambda) * ||grad L_MNC-DPO||_2^2
# ==========================================================
grad_reg_lambda=10.0

training_sample_number=$(wc -l < "${data_path}/train.jsonl")
effective_train_samples=$((training_sample_number - warmup_probe_size))
if [ "${effective_train_samples}" -lt 1 ]; then
  effective_train_samples=1
fi

eval_steps=$(echo "scale=0; ${effective_train_samples} * ${num_train_epochs} / ${bsz} / 4" | bc)
if [ "${eval_steps}" -lt 1 ]; then
  eval_steps=1
fi

case "$opt_alg" in
  ORPO)
    model_name_or_path=$model_path
    ;;
  SLiC)
    loss_type="hinge"
    ;;
  AOT)
    loss_type="aot"
    ;;
  IPO)
    loss_type="ipo"
    beta=0.5
    ;;
  rDPO)
    loss_type="robust"
    ;;
esac

cmd="deepspeed ${deepspeed_args} src/lm_flow/run_peft.py \
  --opt_alg ${opt_alg} \
  --model_name_or_path ${model_name_or_path} \
  --dataset_path ${data_path} \
  --output_dir ${output_dir} --overwrite_output_dir \
  --num_train_epochs ${num_train_epochs} \
  --learning_rate 5e-5 \
  --block_size 512 \
  --per_device_train_batch_size ${bsz} \
  --per_device_eval_batch_size ${bsz} \
  --deepspeed configs/ds_config_zero2.json \
  --bf16 \
  --validation_split_percentage 0 \
  --logging_steps 10 \
  --do_train \
  --use_lora 1 \
  --lora_r 16 \
  --lr_scheduler_type constant \
  --save_steps 999999 \
  --eval_strategy steps \
  --eval_steps ${eval_steps} \
  --weight_decay 0.0001 \
  --dataloader_num_workers 1 \
  --seed 14 \
  --beta ${beta} \
  --loss_type ${loss_type} \
  --enable_margin_threshold \
  --enable_auto_warmup \
  --margin_warmup_steps ${margin_warmup_steps} \
  --warmup_probe_size ${warmup_probe_size} \
  --warmup_probe_batch_size ${warmup_probe_batch_size} \
  --warmup_probe_interval ${warmup_probe_interval} \
  --warmup_ma_window ${warmup_ma_window} \
  --warmup_patience ${warmup_patience} \
  --warmup_max_steps ${warmup_max_steps} \
  --warmup_mc_eps ${warmup_mc_eps} \
  --warmup_mc_min_delta ${warmup_mc_min_delta} \
  --warmup_probe_seed ${warmup_probe_seed} \
  --enable_grad_regularization \
  --grad_reg_lambda ${grad_reg_lambda}"

echo "Running command:"
echo "$cmd"
eval "$cmd"
