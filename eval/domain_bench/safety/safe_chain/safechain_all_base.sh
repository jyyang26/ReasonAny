#!/bin/bash

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- GPU and Parallelism Settings ---
# 定义要使用的GPU设备列表
GPUS=("cuda:5")
# GPUS=("cuda:4" "cuda:5" "cuda:6" "cuda:7")
# 定义每个GPU上允许同时运行的最大模型进程数
MAX_JOBS_PER_GPU=1
# export TORCH_USE_CUDA_DSA=1

# --- Model Class Selection (Uncomment one) ---
# 根据您的要求，使用注释来让您每次只运行一个MODEL_CLASS
# MODEL_CLASS="qwen_14b"
# MODEL_CLASS="qwen_32b_dpsk"
MODEL_CLASS="qwen_32b_qwq"

# --- Path Settings ---
DATASET_PATH="safechain/train-00000-of-00001.parquet"

# 注意：BASE_RESULTS_DIR 会自动包含上面的 $MODEL_CLASS
BASE_RESULTS_DIR="safechain/results/$MODEL_CLASS"

# --- Inference Hyperparameters ---
NUM_SAMPLES=1000      # 从数据集中选择前1000条数据进行推理
BATCH_SIZE=1          # 每个模型的批处理大小（可根据显存调整）
MAX_NEW_TOKENS=4096   # 模型生成的最大token数
TEMPERATURE=0.7       # 生成温度。设为0表示贪心解码
TOP_P=0.9             # Top-p采样

# --- Model List ---
# 更改：将 "models" 定义为关联数组 (associative array)
# 这样我们可以将 (模型名称 -> 模型路径) 映射起来
declare -A models

if [ "$MODEL_CLASS" == "qwen_14b" ]; then
    echo "--- 加载 $MODEL_CLASS 的模型列表 ---"
    models["DeepSeek-R1-Distill-Qwen-14B"]="/models/DeepSeek-R1-Distill-Qwen-14B"
    models["lora_safe_qwen_14b"]="/models/SafeMerge/base_models/lora_safe_qwen_14b"
    models["models--Qwen--Qwen2.5-14B"]="/models/models--Qwen--Qwen2.5-14B"

elif [ "$MODEL_CLASS" == "qwen_32b_dpsk" ]; then
    echo "--- 加载 $MODEL_CLASS 的模型列表 ---"
    models["lora_safe_qwen_32b"]="/models/SafeMerge/base_models/lora_safe_qwen_32b"
    models["models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B"]="/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/b950d47742676362558ae821ef2202f847ac8109"
    models["models--Qwen--Qwen2.5-32B"]="/models/models--Qwen--Qwen2.5-32B/snapshots/1818d35814b8319459f4bd55ed1ac8709630f003"

elif [ "$MODEL_CLASS" == "qwen_32b_qwq" ]; then
    echo "--- 加载 $MODEL_CLASS 的模型列表 ---"
    models["models--Qwen--QwQ-32B"]="/models/models--Qwen--QwQ-32B/snapshots/f28e641280ed3228b25df45b02ce6526b472cbea"

else
    echo "错误：未知的 MODEL_CLASS: $MODEL_CLASS"
    exit 1
fi

set -e

# 计算总共可以并行的任务数量
NUM_GPUS=${#GPUS[@]}
TOTAL_MAX_JOBS=$((NUM_GPUS * MAX_JOBS_PER_GPU))

# 更改：从关联数组获取模型总数
total_models=${#models[@]}

echo "Starting parallel inference..."
echo "Selected MODEL_CLASS: $MODEL_CLASS"
echo "Total models to process: $total_models"
echo "Available GPUs: ${GPUS[*]}"
echo "Max jobs per GPU: $MAX_JOBS_PER_GPU"
echo "Total concurrent jobs: $TOTAL_MAX_JOBS"
echo "------------------------------------------------------"

# 任务启动函数
# 更改：函数现在接受 model_path, model_name, 和 gpu_device
run_task() {
    local model_path=$1
    local model_name=$2
    local gpu_device=$3

    # 更改：不再需要从路径中提取 model_name，它被作为参数$2直接传入
    # local model_name=$(basename "$model_path")
    echo "▶ Launching inference for '$model_name' on device $gpu_device..."
    
    # 更改：output_dir 现在使用您指定的 model_name
    local output_dir="$BASE_RESULTS_DIR/$model_name"
    mkdir -p "$output_dir"
    local output_file="$output_dir/predictions.json"
    
    # 在后台执行Python脚本
    python inference.py \
        --model_path "$model_path" \
        --dataset_path "$DATASET_PATH" \
        --output_path "$output_file" \
        --device "$gpu_device" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" &> "$output_dir/inference.log" &
}

# 循环分发任务
job_count=0
gpu_index=0
# 更改：遍历关联数组的 "键" (即 model_name)
for model_name in "${!models[@]}"; do
    # 检查当前运行的后台任务是否已达到上限
    if [[ $(jobs -r -p | wc -l) -ge $TOTAL_MAX_JOBS ]]; then
        # 等待任何一个后台任务完成
        wait -n
        echo "✓ A job finished. Launching next one."
    fi

    # 以轮询方式为新任务分配GPU
    gpu_device=${GPUS[$gpu_index]}
    
    # 更改：根据 model_name (键) 从数组中获取对应的 model_path (值)
    model_path=${models[$model_name]}
    
    # 启动任务
    # 更改：将 model_path 和 model_name 都传递给函数
    run_task "$model_path" "$model_name" "$gpu_device"
    
    # 更新下一个任务要使用的GPU索引
    gpu_index=$(((gpu_index + 1) % NUM_GPUS))
done

# 等待所有剩余的后台任务完成
echo "------------------------------------------------------"
echo "All models have been launched. Waiting for remaining jobs to complete..."
wait
echo "All inference tasks completed successfully!"