#!/bin/bash

# 確保在遇到錯誤時腳本會退出
set -e


# --- 1. 配置模型和數據路徑 ---
# # qwen
export BASE_MODEL_PATH="PATH TO BASE MODEL"
export SAFETY_MODEL_PATH="PATH TO SAFETY MODEL"
export REASONING_MODEL_PATH="PATH TO REASONING MODEL"

export SAFETY_DATASET_PATH="hh-rlhf/train.parquet"
export REASONING_DATASET_PATH="OpenThoughts-114k-math/data/train-00000-of-00005.parquet"

# --- 2. 配置輸出和超參數 ---
export OUTPUT_DIR="wo_safety_selection"
export DEVICE="cuda:0"
# 用於計算分數的樣本數量
export NUM_SAMPLES=100

# ============== 【Ablation Study 超參數配置】 ==============
# 1. 是否使用 Safety 模型 (True/False)
export IS_USE_SAFETY=False

# 2. 是否使用 Reasoning 模型 (True/False)
export IS_USE_REASONING=True

# 3. 是否使用 Intersection (True/False)
export IS_INTERSECT=False

# 4. 設置重要性分數的計算方法 (True: Gradient, False: SNIP)
export IS_GRADIENT_BASE=True

# 5. 設置 reasoning 模型的選擇策略 (True: Top k, False: Bottom k)
export IS_REASONING_TOP=False
# ==========================================================

# 安全模型權重選擇比例 (Top %)
export SAFETY_RATIO=0.05
# 推理模型權重選擇比例 (% 取決於 IS_REASONING_TOP 的設置)
export REASONING_RATIO=0.05
# 任務向量的縮放因子
export SAFETY_SCALE=1.0
export REASONING_SCALE=1.0

# --- 3. 執行合併腳本 ---
echo "Starting Ablation Study process..."
echo "Base Model: $BASE_MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "--------------------------------------------------"
echo "Ablation Settings:"
echo "  Use Safety Model: $IS_USE_SAFETY"
echo "  Use Reasoning Model: $IS_USE_REASONING"
echo "  Use Intersect: $IS_INTERSECT"
echo "--------------------------------------------------"
echo "Hyperparameters:"
echo "  Score Calculation Method (is_gradient_base): $IS_GRADIENT_BASE"
echo "  Reasoning Selection Strategy (is_reasoning_top): $IS_REASONING_TOP"
echo "--------------------------------------------------"


python /ablation_reasonany.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --safety_model_path "$SAFETY_MODEL_PATH" \
    --reasoning_model_path "$REASONING_MODEL_PATH" \
    --safety_dataset_path "$SAFETY_DATASET_PATH" \
    --reasoning_dataset_path "$REASONING_DATASET_PATH" \
    --output_path "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --safety_ratio $SAFETY_RATIO \
    --reasoning_ratio $REASONING_RATIO \
    --safety_scale $SAFETY_SCALE \
    --reasoning_scale $REASONING_SCALE \
    --is_reasoning_top $IS_REASONING_TOP \
    --is_intersect $IS_INTERSECT \
    --is_gradient_base $IS_GRADIENT_BASE \
    --is_use_safety $IS_USE_SAFETY \
    --is_use_reasoning $IS_USE_REASONING \
    --device "$DEVICE"

echo "Merging process completed."
echo "Final model saved to: $OUTPUT_DIR"