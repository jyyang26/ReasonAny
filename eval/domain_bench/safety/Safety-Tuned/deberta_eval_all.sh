#!/bin/bash

# --- 配置 ---
# 设置模型类别
# export MODEL_CLASS="llama_8b"
export MODEL_CLASS="qwen_14b"

# (*** 已移除 MODEL_LIST ***)
# 脚本现在会自动扫描 $JSON_BASE_DIR 下的所有子目录

# JSON 目录的基础路径
export JSON_BASE_DIR="/safety_eval/$MODEL_CLASS"

# 设置你要运行的Python模块名或脚本路径
export PYTHON_MODULE="evaluation.safety_evals" 

# 设置可用的GPU数量
export NUM_GPUS=3


# --- 核心功能函数 ---
# (*** 此函数保持不变 ***)
# 我们将所有处理逻辑封装在一个函数中
# 这个函数将由父脚本在后台为每个模型调用一次
process_model_directory() {
    # $1 是传递进来的第一个参数 (MODEL_NAME)
    local MODEL_NAME=$1
    local JSON_DIR="${JSON_BASE_DIR}/${MODEL_NAME}"

    # 日志文件的所有输出(echo, python的stdout/stderr)都将进入此函数
    # 而此函数本身在调用时已被重定向到 eval.log

    echo "======================================================"
    echo "后台任务启动于: $(date)"
    echo "开始处理模型: $MODEL_NAME"
    echo "JSON 目录: $JSON_DIR"
    echo "======================================================"

    # 检查目录是否存在 (此检查在父进程中已做过，但双重检查更安全)
    if [ ! -d "$JSON_DIR" ]; then
      echo "错误: 目录 $JSON_DIR 不存在。任务终止。"
      echo "======================================================"
      exit 1 # 退出这个子进程
    fi

    echo "开始并行处理目录中的所有JSON文件..."
    echo "使用 $NUM_GPUS 个GPU..."

    job_count=0
    for json_file in "$JSON_DIR"/*.json; do
      [ -e "$json_file" ] || continue

      gpu_id=$((job_count % NUM_GPUS))
      
      echo "--> 正在将文件 [$(basename "$json_file")] 分配给 GPU:$gpu_id"
      
      # 在后台运行Python脚本
      # 注意：这里的 '&' 只是为了并行化 *此模型中* 的所有json文件
      python -m ${PYTHON_MODULE} --file_path "${json_file}" --device "cuda:${gpu_id}" &
      
      job_count=$((job_count + 1))
    done

    # 'wait' 命令会暂停 *这个子进程*
    # 直到所有为 *当前模型* 启动的python后台任务都执行完毕
    echo ""
    echo "所有 $MODEL_NAME 的Python任务已启动。等待它们全部完成..."
    wait
    echo ""
    echo "✅ 模型 $MODEL_NAME 处理完毕！"
    echo "后台任务结束于: $(date)"
    echo "======================================================"
}

# --- 脚本主体 (父进程) --- (*** 已修改 ***)

# 导出函数，使其对即将创建的子shell(后台进程)可见
export -f process_model_directory

echo "启动后台评估任务..."
echo "======================================================"

# 动态遍历 $JSON_BASE_DIR 下的所有子目录
# '*/' 确保我们只匹配目录
for JSON_DIR_PATH in "$JSON_BASE_DIR"/*/; do
    
    # 检查路径是否确实是一个目录（防止在空目录时 glob 返回字面值）
    if [ ! -d "$JSON_DIR_PATH" ]; then
        echo "!! 警告: 找不到子目录，或 '$JSON_DIR_PATH' 不是一个有效目录。跳过。"
        continue
    fi
    
    # 从完整路径中提取最后一个组件作为 MODEL_NAME
    # (basename 会自动处理结尾的斜杠)
    MODEL_NAME=$(basename "$JSON_DIR_PATH")
    
    # 动态设置日志文件路径
    # $JSON_DIR_PATH 已经包含了结尾的 '/'
    LOG_FILE="${JSON_DIR_PATH}eval.log"

    # 核心步骤：
    # 1. 调用 process_model_directory 函数并传入 MODEL_NAME
    # 2. > "$LOG_FILE" 将标准输出 (stdout) 重定向到日志文件
    # 3. 2>&1 将标准错误 (stderr) 也重定向到标准输出 (即日志文件)
    # 4. & 最后一个 '&' 将整个函数调用放入后台运行
    process_model_directory "$MODEL_NAME" > "$LOG_FILE" 2>&1 &
    
    echo "-> 已为 $MODEL_NAME 启动后台进程。日志将写入: $LOG_FILE"
done

echo "======================================================"
echo "所有后台任务均已启动。父脚本即将退出。"
echo "你可以使用 'tail -f ${JSON_BASE_DIR}/*/eval.log' 来监控所有日志。"
echo "======================================================"