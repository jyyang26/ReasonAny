#!/bin/bash
set -e

# ======================================================================
# 1. 用户配置
# ======================================================================

# --- 核心约束 ---
# 在这里指定您想用于评测的 GPU ID (只允许单个 ID)
GPUS_TO_USE="0"

# 在此 GPU 上同时运行的最大作业（模型）数量
# 您的需求是 "每次只放两个模型"
MAX_CONCURRENT_JOBS=4

# --- 任务定义 (来自您的 opencompass 脚本) ---
MODEL_CLASS="qwen_7b"

DOMAIN_TYPE="finance"


# EVAL TYPE
BASE_MODEL_PATH="/domain_merged_models"
BASE_SAVE_PATH="/domain_merge_eval"

# 要评测的数据集列表
DATASET_LIST=(
    "OpenFinData"
)


MODEL_LIST=(
     "MODEL_NAME"
)

# OpenCompass 的工作目录
# 确保 cd 到 opencompass 目录
cd opencompass || exit 1


# ======================================================================
# 2. 核心脚本逻辑
# ======================================================================

# ---- 辅助函数: 运行单个 OpenCompass 评测任务 ----
# 此函数将在后台被调用
# 参数: $1 = DATASET_NAME, $2 = MODEL_NAME
run_opencompass_eval() {
    local DATASET_NAME=$1
    local MODEL_NAME=$2
    
    local MODEL_HF_PATH="$BASE_MODEL_PATH/$DOMAIN_TYPE/$MODEL_CLASS/$MODEL_NAME"
    local WORK_DIR="$BASE_SAVE_PATH/$DOMAIN_TYPE/$MODEL_CLASS/$EVAL_TYPE/$DATASET_NAME/$MODEL_NAME"
    
    echo "--- [GPU $GPUS_TO_USE] 开始评测: $MODEL_NAME 在 $DATASET_NAME 上 ---"
    echo "    工作目录: $WORK_DIR"
    
    # 确保保存目录存在
    mkdir -p "$WORK_DIR"
    
    # 使用 CUDA_VISIBLE_DEVICES 将此任务锁定到指定的 GPU
    # 注意：这里的 nohup 和重定向是评测命令的一部分
    CUDA_VISIBLE_DEVICES=$GPUS_TO_USE nohup opencompass \
        --hf-path "$MODEL_HF_PATH" \
        --hf-type base \
        --datasets "${DATASET_NAME}_gen" \
        --work-dir "$WORK_DIR" \
        --max-num-worker 1 \
        --max-out-len 2048 \
        > "$WORK_DIR/evaluation.log" 2>&1
        
    echo "--- [GPU $GPUS_TO_USE] 完成评测: $MODEL_NAME 在 $DATASET_NAME 上 ---"
}

# 导出函数，使其在子 shell 中可用
export -f run_opencompass_eval
export GPUS_TO_USE MODEL_CLASS BASE_MODEL_PATH BASE_SAVE_PATH


# ---- 主逻辑: 使用作业控制来管理并发 ----

echo "启动评测..."
echo "使用 GPU: $GPUS_TO_USE"
echo "最大并发任务数: $MAX_CONCURRENT_JOBS"

# 遍历每个数据集（串行）
for DATASET_NAME in "${DATASET_LIST[@]}"; do
    echo "========================================================="
    echo "=== 开始处理数据集: $DATASET_NAME"
    echo "========================================================="
    
    # 遍历此数据集的所有模型（并行，但受控）
    for MODEL_NAME in "${MODEL_LIST[@]}"; do
        
        # 1. 在后台启动评测任务
        echo "--> 正在启动: $MODEL_NAME..."
        run_opencompass_eval "$DATASET_NAME" "$MODEL_NAME" &
        
        # 2. 检查当前运行的作业数量
        # 'jobs -p' 列出所有后台作业的 PID
        # 'wc -l' 统计有多少行 (即多少个作业)
        current_jobs=$(jobs -p | wc -l)
        echo "    当前运行的作业数: $current_jobs"

        # 3. 如果达到最大并发数，则等待
        if [[ "$current_jobs" -ge "$MAX_CONCURRENT_JOBS" ]]; then
            echo "--- 达到最大并发数 ($MAX_CONCURRENT_JOBS)，等待一个任务完成... ---"
            # 'wait -n' 会等待任何一个后台作业完成
            wait -n
            echo "--- 一个任务已完成，继续... ---"
        fi
    done
    
    # 4. 单个数据集循环结束，等待该数据集所有剩余的作业完成
    echo "--- $DATASET_NAME 的所有模型均已启动，正在等待剩余任务完成... ---"
    wait
    echo "=== 数据集 $DATASET_NAME 全部处理完毕 ==="
done

echo "--- 所有评测任务已全部完成 ---"