#!/bin/bash

# 设置训练超参数
NEO4J_URI="neo4j://127.0.0.1:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="12345678"
SAMPLE_SIZE="None"  # 设置为 None 表示使用所有数据
EPOCHS=50
LEARNING_RATE=0.001
SAVE_DIR="../training/models/link_prediction_v4"
HIDDEN_DIM=128
OUT_DIM=384
NUM_HEADS=8

# 运行训练脚本
echo "Start training..."
python ../han_model.py \
    "$NEO4J_URI" \
    "$NEO4J_USERNAME" \
    "$NEO4J_PASSWORD" \
    "$SAMPLE_SIZE" \
    "$EPOCHS" \
    "$LEARNING_RATE" \
    "$SAVE_DIR" \
    "$HIDDEN_DIM" \
    "$OUT_DIM" \
    "$NUM_HEADS"