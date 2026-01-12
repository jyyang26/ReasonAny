#!/bin/bash

# 确保Python脚本的名称与您保存的一致
PYTHON_SCRIPT="openai_eval_chain.py"

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi

echo "Starting parallel processing for 4 directories..."

# 1. 定义要处理的目录数组
# 注意：路径末尾不要加 /
BASE_PATH="/qwen_32b_qwq"


# qwq 32
DIRS=(
    "$BASE_PATH/MODEL_NAME"
)


# 2. 循环并启动后台任务
for DIR in "${DIRS[@]}"; do
    if [ ! -d "$DIR" ]; then
        echo "Warning: Directory not found, skipping: $DIR"
        continue
    fi
    
    echo "Spawning process for: $DIR"
    
    # --- 修改开始 ---
    
    # 1. 定义 rating_chain 目录路径
    RATING_DIR="$DIR/rating_chain"
    
    # 2. 确保该目录存在 (这对于重定向日志文件至关重要)
    #    -p 选项确保如果目录已存在也不会报错
    mkdir -p "$RATING_DIR"
    
    # 3. 定义日志文件的新路径
    #    这将捕获Python脚本的所有 print() 输出和错误信息
    LOG_FILE="$RATING_DIR/rating_process.log"
    
    echo "Log file will be written to: $LOG_FILE"
    
    # 4. 使用 python3 运行脚本，并将目录作为参数传递
    #    使用 & 将其放入后台
    #    将标准输出(stdout)和标准错误(stderr)都重定向到新的日志文件
    python3 "$PYTHON_SCRIPT" "$DIR" > "$LOG_FILE" 2>&1 &
    
    # --- 修改结束 ---
    
done

# 3. 等待所有后台任务完成
echo "All processes spawned. Waiting for completion... (This may take a long time)"
wait

echo "==================================="
echo "All processes finished."
echo "Check 'log_*.log' files for detailed output from each process."
echo "Check 'rating_chain/predictions_with_ratings.json' in each directory for results."