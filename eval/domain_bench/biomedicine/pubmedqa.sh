#!/bin/bash

# --- 1. 配置 ---

# !! 在此处手动设定每个 GPU 的最大并行模型数
MAX_MODELS_PER_GPU=4



MODEL_LIST=(
    "MODEL_NAME"
)

# 使用的 GPU 列表
GPUS=(1 2)

# Python 脚本的路径
PYTHON_SCRIPT_PATH="../domain_exp/experiment_pubmedqa.py"

# --- 2. 路径配置 ---
MODEL_CLASS="qwen_7b"
BASE_MODEL_DIR="domain_merged_models/biomedicine"
BASE_SAVE_DIR="domain_merge_eval/biomedicine"

# --- 3. 动态分配模型列表 ---
# 采用轮询 (Round-Robin) 方式将模型分配给每个 GPU
LIST_GPU0=()
LIST_GPU1=()

num_gpus=${#GPUS[@]}
i=0
for model in "${MODEL_LIST[@]}"; do
    gpu_index=$(( i % num_gpus ))
    if (( gpu_index == 0 )); then
        LIST_GPU0+=("$model")
    else
        LIST_GPU1+=("$model")
    fi
    i=$(( i + 1 ))
done

echo "将在 GPU 0 (共 ${#LIST_GPU0[@]} 个) 上运行: ${LIST_GPU0[*]}"
echo "将在 GPU 1 (共 ${#LIST_GPU1[@]} 个) 上运行: ${LIST_GPU1[*]}"
echo "每个 GPU 的最大并行数设置为: $MAX_MODELS_PER_GPU"
echo "---"


# --- 4. 定义一个函数来运行单个模型 ---
# 这个函数将被 xargs 调用
run_model() {
    DEVICE_NUMBER=$1
    MODEL_NAME=$2
    
    # 从外部环境变量中读取
    # (我们稍后会 export 它们)

    # 构造动态路径
    MODEL_PATH="${BASE_MODEL_DIR}/${MODEL_CLASS}/${MODEL_NAME}"
    SAVE_PATH="${BASE_SAVE_DIR}/${MODEL_CLASS}/domain_eval/Medical/PubMedQA/${MODEL_NAME}/results.json"
    
    # 创建保存目录
    mkdir -p "$(dirname "$SAVE_PATH")"
    
    LOG_FILE="${MODEL_NAME}_gpu${DEVICE_NUMBER}.log"
    
    echo "正在启动: ${MODEL_NAME} on GPU ${DEVICE_NUMBER}. 日志: ${LOG_FILE}"
    
    # 执行 Python 脚本
    python ${PYTHON_SCRIPT_PATH} \
        --model_name "${MODEL_NAME}" \
        --device_number ${DEVICE_NUMBER} \
        --model_path "${MODEL_PATH}" \
        --save_path "${SAVE_PATH}" \
        --model_class "${MODEL_CLASS}" > "${BASE_SAVE_DIR}/${MODEL_CLASS}/domain_eval/Medical/PubMedQA/${MODEL_NAME}/${LOG_FILE}" 2>&1
    
    echo "已完成: ${MODEL_NAME} on GPU ${DEVICE_NUMBER}"
}

# 导出函数和变量，以便 xargs 的子 shell 可以访问它们
export -f run_model
export PYTHON_SCRIPT_PATH MODEL_CLASS BASE_MODEL_DIR BASE_SAVE_DIR

# --- 5. 并行执行 ---

# 启动 GPU 0 的作业池
# 修改说明：将 bash -c 后的单引号改为双引号 "..."，并在 {} 周围加单引号 '{}'
if (( ${#LIST_GPU0[@]} > 0 )); then
    printf "%s\n" "${LIST_GPU0[@]}" | xargs -P $MAX_MODELS_PER_GPU -I {} \
        bash -c "run_model ${GPUS[0]} '{}'" &
fi
PID_GPU0=$! 

# 启动 GPU 1 的作业池
# 修改说明：同上，使用双引号允许父 Shell 解析 ${GPUS[1]}
if (( ${#LIST_GPU1[@]} > 0 )); then
    printf "%s\n" "${LIST_GPU1[@]}" | xargs -P $MAX_MODELS_PER_GPU -I {} \
        bash -c "run_model ${GPUS[1]} '{}'" &
fi
PID_GPU1=$!

# --- 6. 等待所有作业完成 ---
echo "---"
echo "所有作业已启动。"
echo "GPU 0 作业池 PID: $PID_GPU0"
echo "GPU 1 作业池 PID: $PID_GPU1"
echo "等待所有模型处理完成... (日志将被写入 .log 文件)"

# 等待两个 xargs 进程 (作业池) 都完成
wait $PID_GPU0
wait $PID_GPU1

echo "---"
echo "所有作业已完成。"