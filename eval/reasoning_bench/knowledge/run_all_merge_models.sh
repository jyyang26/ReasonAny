

cd /opencompass
DATASET_LIST=(
    "ARC_e"
    "ARC_c"
    "mmlu_llm_judge"
    "gpqa_llm_judge"
    "humaneval"
    "livecodebench"
)

MODEL_CLASS="qwen_7b"
# or you can choose "qwen_14b" "qwen_32b" "llama_8b"

MODEL_LIST=(
    "LIST ALL MODEL NAMES"
)

# 循环遍历模型列表，为每个模型启动一个后台评估任务
for DATASET_NAME in "${DATASET_LIST[@]}"
do
    for MODEL_NAME in "${MODEL_LIST[@]}"
    do
        # 为当前模型设置独立的工作目录
        WORK_DIR="/PATH_TO_YOUR_OUTPUT_DIR/$DATASET_NAME/$MODEL_CLASS/$MODEL_NAME"

        # 打印提示信息，方便追踪
        echo "Starting evaluation for model: $MODEL_NAME"
        echo "Log file will be saved in: $WORK_DIR/evaluation.log"

        mkdir -p "$WORK_DIR" && CUDA_VISIBLE_DEVICES=0 nohup opencompass \
            --hf-path $PATH_TO_YOUR_DIR/$MODEL_CLASS/$MODEL_NAME \
            --hf-type base \
            --datasets "${DATASET_NAME}_gen" \
            --work-dir "$WORK_DIR" \
            --max-num-worker 1 \
            --max-out-len 2048 \
            > "$WORK_DIR/evaluation.log" 2>&1 &
    done
done

echo "All evaluation tasks have been launched in the background."
echo "You can use 'jobs' or 'ps -ef | grep opencompass' to check their status."