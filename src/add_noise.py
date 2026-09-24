import argparse
import json
import os
import random


def add_noise_to_split(
    input_jsonl_path: str,
    output_pref_jsonl_path: str,
    output_sft_json_path: str,
    split_name: str,
    noise_rate: float,
    seed: int = 42,
):
    """
    对一个 split（通常是 train）添加噪声，并同时写出：
      1. 偏好数据 (RLHF)：output_pref_jsonl_path
      2. SFT 数据：output_sft_json_path

    假定 input_jsonl_path 是 jsonl，每行格式：
      {"prompt": ..., "chosen": ..., "rejected": ...}
    """
    if not os.path.exists(input_jsonl_path):
        print(f"[WARN] {input_jsonl_path} 不存在，跳过该 split。")
        return

    os.makedirs(os.path.dirname(output_pref_jsonl_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_sft_json_path), exist_ok=True)

    random.seed(seed)

    cnt_total = 0
    cnt_noise = 0
    sft_instances = []

    with open(input_jsonl_path, "r", encoding="utf-8") as fin, \
            open(output_pref_jsonl_path, "w", encoding="utf-8") as fp_out:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            prompt = data["prompt"]
            chosen = data["chosen"]
            rejected = data["rejected"]

            cnt_total += 1

            # ====== 噪声逻辑 ======
            if split_name == "train" and random.random() < noise_rate:
                # 有噪声：交换 pair 顺序，并用 rejected 作为 sft_target
                responses = [rejected, chosen]
                sft_target = rejected
                cnt_noise += 1
            else:
                # 无噪声：保持正常顺序，用 chosen 作为 sft_target
                responses = [chosen, rejected]
                sft_target = chosen

            # 写偏好数据（无论是否加噪）
            json.dump(
                {
                    "prompt": prompt,
                    "chosen": responses[0],
                    "rejected": responses[1],
                },
                fp_out,
                ensure_ascii=False,
            )
            fp_out.write("\n")

            # 收集 SFT 数据
            sft_instances.append(
                {
                    "text": f"\n\nHuman: {prompt}\n\nAssistant: {sft_target}"
                }
            )

    # 写 SFT json
    sft_obj = {
        "type": "text_only",
        "instances": sft_instances,
    }
    with open(output_sft_json_path, "w", encoding="utf-8") as fsft:
        json.dump(sft_obj, fsft, ensure_ascii=False)

    print(
        f"[{split_name}] 总样本 {cnt_total} 条，其中加入噪声 {cnt_noise} 条 "
        f"(noise_rate={noise_rate})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录，里面至少包含 train.jsonl（由 data_cleaning.py 生成）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录，用于保存带噪声的偏好数据和 SFT 数据",
    )
    parser.add_argument(
        "--noise_rate",
        type=float,
        default=0.1,
        help="对 train split 加噪的概率（交换 chosen / rejected）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证可复现",
    )

    args = parser.parse_args()

    # 针对 train split
    input_train = os.path.join(args.input_dir, "train.jsonl")
    output_train_pref = os.path.join(args.output_dir, "train.jsonl")
    output_train_sft = os.path.join(args.output_dir, "sft", "train.json")

    add_noise_to_split(
        input_jsonl_path=input_train,
        output_pref_jsonl_path=output_train_pref,
        output_sft_json_path=output_train_sft,
        split_name="train",
        noise_rate=args.noise_rate,
        seed=args.seed,
    )

    # 如果你未来有 val / test 也想处理，可以按需再调用一次：
    # input_val = os.path.join(args.input_dir, "val.jsonl")
    # output_val_pref = os.path.join(args.output_dir, "val.jsonl")
    # output_val_sft = os.path.join(args.output_dir, "sft", "val.json")
    # add_noise_to_split(
    #     input_jsonl_path=input_val,
    #     output_pref_jsonl_path=output_val_pref,
    #     output_sft_json_path=output_val_sft,
    #     split_name="val",   # val/test 一般不加噪声，除非你特意想要
    #     noise_rate=args.noise_rate,
    #     seed=args.seed,
    # )


if __name__ == "__main__":
    main()
