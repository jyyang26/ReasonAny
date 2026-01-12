#!/bin/bash
set -e



# qwen
export BASE_MODEL_PATH="PATH TO BASE MODEL"
export SAFETY_MODEL_PATH="PATH TO SAFETY MODEL"
export REASONING_MODEL_PATH="PATH TO REASONING MODEL"

export SAFETY_DATASET_PATH="hh-rlhf/train.parquet"
export REASONING_DATASET_PATH="OpenThoughts-114k-math/data/train-00000-of-00005.parquet"

export OUTPUT_DIR="MERGED MODEL"
export DEVICE="cuda:0"

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
export IS_INTERSECT=False
# ==========================================

# 安全模型權重選擇比例 (Top %)
export SAFETY_RATIO=0.05
# 推理模型權重選擇比例 (% 取決於 IS_REASONING_TOP 的設置)
export REASONING_RATIO=0.05
# 任務向量的縮放因子
export SAFETY_SCALE=1.0
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
    --device "$DEVICE"

echo "Merging process completed."
echo "Final model saved to: $OUTPUT_DIR"