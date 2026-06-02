#!/bin/bash

# 创建基础目录
mkdir -p /data/dengyan/datasets

# 1. 下载 table-benchmark/tqabench
hf download table-benchmark/tqabench \
	  --repo-type dataset \
	    --local-dir /data/dengyan/datasets/tqabench

# 2. 下载 DongfuJiang/FeTaQA
hf download DongfuJiang/FeTaQA \
	  --repo-type dataset \
	    --local-dir /data/dengyan/datasets/FeTaQA

# 3. 下载 Multilingual-Multimodal-NLP/TableBench
hf download Multilingual-Multimodal-NLP/TableBench \
	  --repo-type dataset \
	    --local-dir /data/dengyan/datasets/TableBench
