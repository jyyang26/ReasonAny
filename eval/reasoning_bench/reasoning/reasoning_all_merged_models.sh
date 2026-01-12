#!/bin/bash

# 激活 Conda 环境

# --- 配置 ---
PROMPT_TYPE="deepseek-math"

DATASET="gsm8k"
# DATASET="math500" 
# DATASET="aime24"
SPLIT="test"
NUM_TEST_SAMPLE=-1 # -1 表示使用所有测试样本




MODEL_LIST=(
    "MODEL NAMES"
)

# GPU设备列表
GPU_LIST=(0)

MODEL_CLASS_LIST=("qwen_7b") # 要运行的模型类别列表

DOMAIN_TYPE="finance"

# --- 并发池设置 ---
NUM_GPUS=${#GPU_LIST[@]}

# 检查GPU列表是否为空
if [ $NUM_GPUS -eq 0 ]; then
    echo "错误：GPU_LIST 不能为空！"
    exit 1
fi

# 1. 创建一个FIFO（命名管道）来管理GPU
#    使用 $$ 确保FIFO名称在并发脚本中是唯一的
FIFO_PATH="/tmp/gpu_fifo_$$"
mkfifo "$FIFO_PATH"

# 2. 将FIFO与文件描述符 3 关联，以便读写
exec 3<>"$FIFO_PATH"

# 3. 注册一个陷阱 (trap)，以确保脚本退出时（即使是异常退出）FIFO文件被删除
trap 'rm -f "$FIFO_PATH"' EXIT

# 4. "填充" FIFO：将所有可用的GPU ID作为令牌放入管道
for GPU_ID in "${GPU_LIST[@]}"; do
    echo "$GPU_ID" >&3
done

echo "使用 ${NUM_GPUS} 个GPU作为并发工作池: ${GPU_LIST[*]}"
echo "---"


# --- 循环遍历并为每个模型启动一个 *并发* 评估任务 ---

# 外层循环：遍历所有模型类别
for MODEL_CLASS in "${MODEL_CLASS_LIST[@]}"; do

    echo "######################################################"
    echo "### 开始排队 MODEL_CLASS: $MODEL_CLASS"
    echo "######################################################"

    # 内层循环：遍历该类别的所有模型
    for i in "${!MODEL_LIST[@]}"; do
        MODEL_NAME=${MODEL_LIST[$i]}

        # 5. 从FIFO读取一个可用的GPU ID
        #    如果管道是空的，这个命令将会阻塞，直到有GPU被释放
        read -u 3 GPU_ID
        
        # 6. 一旦获取到GPU ID，就在后台启动一个子shell来执行任务
        (
            # --- 子shell内部 ---
            # 这里的代码将在后台执行，使用分配到的 $GPU_ID

           
            # domain output
            WORK_DIR="OUTPUT_DIR/${DOMAIN_TYPE}/${MODEL_CLASS}/performance_eval/${DATASET}/$MODEL_NAME"
            
            # 创建输出目录
            mkdir -p "$WORK_DIR"
            
            echo "===================================================="
            echo ">>> [GPU $GPU_ID] 任务启动: '$MODEL_NAME' [CLASS: $MODEL_CLASS]"
            echo "    模型路径: $MODEL_PATH"
            echo "    日志文件: $WORK_DIR/evaluation.log"
            echo "===================================================="

            # 设置当前任务可见的GPU设备并启动评估脚本
            # 所有输出 (stdout 和 stderr) 都将被重定向到 evaluation.log 文件
            CUDA_VISIBLE_DEVICES=$GPU_ID TOKENIZERS_PARALLELISM=false python3 -u /Qwen2.5-Math/evaluation/math_eval.py \
                --model_name_or_path "${MODEL_PATH}" \
                --data_name "${DATASET}" \
                --output_dir "${WORK_DIR}" \
                --split "${SPLIT}" \
                --prompt_type "${PROMPT_TYPE}" \
                --num_test_sample "${NUM_TEST_SAMPLE}" \
                --seed 0 \
                --temperature 0 \
                --n_sampling 1 \
                --top_p 1 \
                --start 0 \
                --end -1 \
                --save_outputs \
                --overwrite \
                --use_vllm > "$WORK_DIR/evaluation.log" 2>&1

            # 检查上一个命令是否成功执行
            if [ $? -ne 0 ]; then
                echo "XXX [GPU $GPU_ID] 任务失败: '$MODEL_NAME' [CLASS: $MODEL_CLASS]"
                echo "    请检查日志: $WORK_DIR/evaluation.log"
            else
                echo "<<< [GPU $GPU_ID] 任务完成: '$MODEL_NAME' [CLASS: $MODEL_CLASS]"
            fi
            echo "----------------------------------------------------"

            # 7. 任务完成（无论成功与否），将GPU ID归还到FIFO管道
            echo "$GPU_ID" >&3
            
        ) & # 将整个子shell ( ... ) 放入后台执行

    done # 内层 (MODEL_LIST) 循环结束
done # 外层 (MODEL_CLASS_LIST) 循环结束

# 8. 等待所有在后台启动的子shell任务全部完成
echo
echo "######################################################"
echo "### 所有任务均已启动。等待所有后台进程执行完毕..."
echo "######################################################"
wait

# 9. 关闭文件描述符
exec 3>&-

echo "All evaluation tasks have been completed."
# 'trap' 将在此处自动执行，清理FIFO文件