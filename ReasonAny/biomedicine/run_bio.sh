#!/bin/bash

# 確保在遇到錯誤時腳本會退出
set -e


export BASE_MODEL_PATH="PATH TO BASE MODEL"
export DOMAIN_MODEL_PATH="PATH TO DOMAIN MODEL"
export REASONING_MODEL_PATH="PATH TO REASONING MODEL"
export DOMAIN_DATASET_PATH="pubmedqa/train-00000-of-00001.parquet"
export REASONING_DATASET_PATH="OpenThoughts-114k-math/data/train-00000-of-00005.parquet"

# --- 2. 配置輸出和超參數 ---
export OUTPUT_DIR="MERGED MODEL"
export DEVICE="cuda:0"
# 用於計算分數的樣本數量
export NUM_SAMPLES=100

# ============== 【超參數配置】 ==============
# 設置重要性分數的計算方法
# 設置為 True:  使用梯度幅值 |g| (Gradient)
# 設置為 False: 使用 SNIP 分數 |w*g|
export IS_GRADIENT_BASE=True

# 設置 reasoning 模型的選擇策略
# 設置為 True:  選擇分數【最高】的權重 (top k)
# 設置為 False: 選擇分數【最低】的權重 (bottom k)
export IS_REASONING_TOP=False
# ==========================================

# 安全模型權重選擇比例 (Top %)
export DOMAIN_RATIO=0.05
# 推理模型權重選擇比例 (% 取決於 IS_REASONING_TOP 的設置)
export REASONING_RATIO=0.05
# 任務向量的縮放因子
export DOMAIN_SCALE=1.0
export REASONING_SCALE=1.0

# --- 3. 執行合併腳本 ---
echo "Starting ReasonAny process..."
echo "Base Model: $BASE_MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "--------------------------------------------------"
echo "Hyperparameters:"
echo "  Score Calculation Method (is_gradient_base): $IS_GRADIENT_BASE"
echo "  Reasoning Selection Strategy (is_reasoning_top): $IS_REASONING_TOP"
echo "--------------------------------------------------"


python main.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --domain_model_path "$DOMAIN_MODEL_PATH" \
    --reasoning_model_path "$REASONING_MODEL_PATH" \
    --domain_dataset_path "$DOMAIN_DATASET_PATH" \
    --reasoning_dataset_path "$REASONING_DATASET_PATH" \
    --output_path "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --domain_ratio $DOMAIN_RATIO \
    --reasoning_ratio $REASONING_RATIO \
    --domain_scale $DOMAIN_SCALE \
    --reasoning_scale $REASONING_SCALE \
    --is_reasoning_top $IS_REASONING_TOP \
    --is_gradient_base $IS_GRADIENT_BASE \
    --device "$DEVICE"

echo "Merging process completed."
echo "Final model saved to: $OUTPUT_DIR"