#!/bin/bash

# 设置脚本参数
INPUT_FILE="../data/raw/data.jsonl"
PROCESSED_DIR="../data/processed"
CSV_PATH="../data/processed/papers.csv"

echo "Processing the raw .jsonl file..."
python ../data/process_data.py --input_file "$INPUT_FILE" --processed_dir "$PROCESSED_DIR"

# 运行时间分布分析脚本
echo "Analyzing paper distribution..."
python ../data/analyze_time_distribution.py --csv_path "$CSV_PATH"