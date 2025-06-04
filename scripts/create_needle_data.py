# import os
# import argparse
# import json
# from tqdm import tqdm
# from datasets import load_dataset

# parser = argparse.ArgumentParser()
# parser.add_argument("--output_path", type=str, default="data/pg19.jsonl")
# args = parser.parse_args()

# os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

# dset = load_dataset("pg19")["train"]
# with open(args.output_path, "w") as f:
#     for elem in tqdm(dset):
#         data = {"text": elem["text"]}
#         f.write(f"{json.dumps(data)}\n")
import os
import json
from datasets import load_dataset
from tqdm import tqdm

# 加载前 1000 个样本
dataset = load_dataset("deepmind/pg19", trust_remote_code=True, split="train[:1000]")

# 创建保存目录
os.makedirs("data", exist_ok=True)

# 保存为 JSONL 文件
with open("data/pg19_subset.jsonl", "w", encoding="utf-8") as f:
    for example in tqdm(dataset):
        json.dump({"text": example["text"]}, f, ensure_ascii=False)
        f.write("\n")
