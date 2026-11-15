# DACS-DPO: reproduction commands

## Environment and placement

Environment used: Python 3.9.25, PyTorch 2.8.0+cu129, Transformers 4.57.6, TRL 0.19.0, PEFT 0.10.0, Accelerate 1.10.1, Datasets 4.5.0, DeepSpeed 0.15.4, bitsandbytes 0.48.2, OpenAI 2.8.0, editable LMFlow 1.0.0, and eight RTX 4090 GPUs (48GB).

| File in this package | Place in the existing project |
| --- | --- |
| `generate_preference_noise.py` | `src/generate_preference_noise.py` |
| `train_dacsdpo.py` | `src/lm_flow/train_dacsdpo.py` |
| `evaluate_win_rate.py` | `src/lm_test/evaluate_win_rate.py` |
| `run_dacsdpo.sh` | `scripts/run_dacsdpo.sh` |

## Noise generation

```bash
python src/generate_preference_noise.py \
  --input_dir datasets/UltraFeedback/no_clean \
  --output_dir datasets/UltraFeedback/no_clean_0.4 \
  --noise_rate 0.4 --seed 42
```

Output: `datasets/UltraFeedback/no_clean_0.4/train.jsonl` and `noise_manifest.json`.

## Training

Arguments: method, training seed, patience. 
```bash
bash scripts/run_dacsdpo.sh dacs 14 50
```

Other available methods: `dpo`, `dac`, `dpo_regularizer`. Replace `dacs` in the command to run one of them. Outputs are under `models/UltraFeedback/<method>_release_uf40_seed14/`; 

## Evaluation

Preference accuracy (PA):

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  src/lm_test/test_peft.py --opt_alg DPO \
  --model_name_or_path models/UltraFeedback/sft_DPO_pythia2_no_clean \
  --dataset_path datasets/UltraFeedback/no_clean_0.4 \
  --output_dir models/UltraFeedback/dacs_release_uf40_seed14 \
  --per_device_eval_batch_size 16 \
  --deepspeed configs/ds_config_zero3_for_eval.json --bf16 \
  --beta 0.01 --loss_type sigmoid --seed 14
```

Win rate (WR): set your judge endpoint and key as environment variables. 
```bash
export JUDGE_API_BASE_URL="https://YOUR_PROVIDER/v1"
export JUDGE_API_KEY="YOUR_KEY"
export JUDGE_API_STYLE=responses
export JUDGE_REASONING_EFFORT=xhigh
python src/lm_test/evaluate_win_rate.py \
  --dataset_path datasets/UltraFeedback/no_clean \
  --model_root models/UltraFeedback \
  --model_name_or_path models/UltraFeedback/sft_DPO_pythia2_no_clean \
  --base_model_path EleutherAI/pythia-2.8b \
  --model_names dacs_release_uf40_seed14 \
  --output_dir results/UltraFeedback/dacs_release_uf40_seed14 \
  --n_prompts 99 --seed 14 --gpu_mode single_gpu \
  --quantization 4bit --max_input_tokens 512 --max_new_tokens 512 \
  --judge_model gpt-**-** --judge_workers 8
```

