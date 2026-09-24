#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4
deepspeed_args="--master_port=17901 --include=localhost:0,1"

# Parses arguments
model_name_or_path=meta-llama/Meta-Llama-3-8B
dataset=AnthropicHH
version=no_clean
pretty_name=dpo_llama3_8b_no_clean

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset)
      dataset="$2"
      shift
      ;;
    --version)
      version="$2"
      shift
      ;;
    --pretty_name)
      pretty_name="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

dataset_path=datasets/${dataset}/${version}/sft
output_dir=models/${dataset}/sft_${pretty_name}

deepspeed ${deepspeed_args} \
  src/lm_flow/run_sft.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --disable_group_texts 1 \
    --block_size 512 \
    --per_device_train_batch_size 12\
    --deepspeed configs/ds_config_zero2.json \
    --bf16 \
    --run_name finetune \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 99999 \
    --dataloader_num_workers 1