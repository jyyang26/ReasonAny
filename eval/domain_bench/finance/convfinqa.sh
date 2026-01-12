#!/bin/bash

# --- 1. 配置 ---

# !! 每一个显卡上并行的模型数量
MAX_NUM_MODEL=2

# 指定使用的 GPU (这里只使用 gpu:0)
TARGET_GPU=0

MODEL_LIST=(
    "MODEL_NAME"
)

# Python 脚本路径
PYTHON_SCRIPT_PATH="../domain_exp/experiment_convfinqa.py"

# --- 2. 路径配置 ---
model_class="qwen_7b"
DOMAIN_TYPE="finance"
BASE_MODEL_DIR="domain_merged_models/${DOMAIN_TYPE}"

# 结果保存的基础路径 (更新为您要求的新路径)
BASE_SAVE_DIR="domain_merge_eval/finance/${model_class}/domain_eval/finance/ConvfinQA"

# --- 3. 定义运行函数 ---
run_model() {
    MODEL_NAME=$1
    
    # 构造完整路径
    MODEL_PATH="${BASE_MODEL_DIR}/${model_class}/${MODEL_NAME}"
    
    # 构造保存路径: .../ConvfinQA/{model_name}/results.json
    SAVE_PATH="${BASE_SAVE_DIR}/${MODEL_NAME}/results.json"
    mkdir -p "$(dirname "$SAVE_PATH")"
    # 创建日志文件名
    LOG_FILE="${MODEL_NAME}_convfinqa.log"
    
    echo "正在启动: ${MODEL_NAME} on GPU ${TARGET_GPU}..."
    echo "  模型: ${MODEL_PATH}"
    echo "  保存: ${SAVE_PATH}"
    
    # 执行 Python 脚本
    python ${PYTHON_SCRIPT_PATH} \
        --model_name "${MODEL_NAME}" \
        --device_number ${TARGET_GPU} \
        --model_path "${MODEL_PATH}" \
        --save_path "${SAVE_PATH}" > "${BASE_SAVE_DIR}/${MODEL_NAME}/${LOG_FILE}" 2>&1
    
    echo "已完成: ${MODEL_NAME}"
}

# 导出变量和函数供 xargs 使用
export -f run_model
export PYTHON_SCRIPT_PATH BASE_MODEL_DIR BASE_SAVE_DIR TARGET_GPU model_class

# --- 4. 并行执行 ---
echo "--- 开始执行 ConvFinQA 评估 ---"
echo "GPU: ${TARGET_GPU}"
echo "并行数: ${MAX_NUM_MODEL}"
echo "模型列表: ${MODEL_LIST[*]}"

# 使用 xargs 控制并行度
# -P: 最大进程数
# -I {}: 替换符号
if (( ${#MODEL_LIST[@]} > 0 )); then
    printf "%s\n" "${MODEL_LIST[@]}" | xargs -P $MAX_NUM_MODEL -I {} \
        bash -c 'run_model "{}"'
fi

echo "--- 所有任务已提交，请检查日志文件 (.log) ---"