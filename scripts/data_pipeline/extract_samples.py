# 路径: scripts/data_pipeline/extract_samples.py
# 用法: python extract_samples.py --dataset bird --num 3
import json
import os
import random
import argparse

# 数据集配置字典 (泛化的核心)
# 格式: "数据集名称": ("相对路径/原始JSON", "相对路径/输出样例JSON")
DATASET_CONFIGS = {
    "spider": (
        "spider/spider/train_spider.json",
        "spider/samples/spider_sample.json"
    ),
    "bird": (
        "bird/train/train.json",
        "bird/samples/bird_sample.json"
    )
    # 未来可以轻松扩展，例如：
    # "wikisql": ("wikisql/train.json", "wikisql/samples/wikisql_sample.json")
}

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_sample(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"🌟 成功提取并保存至: {output_path}")

def extract_from_dataset(dataset_name, base_dir, num_samples):
    if dataset_name not in DATASET_CONFIGS:
        print(f"❌ 错误: 未知的数据集 '{dataset_name}'")
        return

    rel_input, rel_output = DATASET_CONFIGS[dataset_name]
    input_file = os.path.join(base_dir, rel_input)
    output_file = os.path.join(base_dir, rel_output)

    if not os.path.exists(input_file):
        print(f"⚠️ 警告: 找不到输入文件 {input_file}，请先运行下载脚本。")
        return

    data = load_json(input_file)
    samples = random.sample(data, min(num_samples, len(data)))
    save_sample(samples, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="抽取多表数据集的样本 JSON")
    parser.add_argument("--dataset", type=str, default="all", help="指定数据集名称 (spider, bird, all)")
    parser.add_argument("--num", type=int, default=3, help="提取的样本数量")
    parser.add_argument("--base_dir", type=str, default="/data/dengyan/datasets", help="数据的根目录")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASET_CONFIGS.keys():
            extract_from_dataset(ds, args.base_dir, args.num)
    else:
        extract_from_dataset(args.dataset, args.base_dir, args.num)
