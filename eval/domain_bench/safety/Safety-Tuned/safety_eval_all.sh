#!/bin/bash

# 设置脚本在遇到错误时立即退出
set -e
# export MODEL_CLASS="llama_8b"
export MODEL_CLASS="qwen_7b"
# export MODEL_CLASS="qwen_14b"
# export MODEL_CLASS="qwen_32b_qwq"
# 定义模型名称列表
export MODEL_LIST=(
    "MODEL_NAME"
)

export GPUS=(1)
# export GPUS=(1 3)

# ==================== 新增配置 ====================
# 设置每张卡上允许并行的最大模型（或作业）数量
export MAX_MODEL_PER_CUDA=4
# =================================================


# --- 静态配置区 (不随模型变化) ---
# Python 脚本的路径
PYTHON_SCRIPT="/safety-tuned-llamas/generation/inference.py"

# 输入文件夹路径
# INPUT_DIR="/safety-tuned-llamas/data/evaluation/eval_sub_1"
INPUT_DIR="/safety-tuned-llamas/data/evaluation/eval_all"
# 默认的 prompt 模板路径
PROMPT_TEMPLATE="/safety-tuned-llamas/configs/alpaca.json" 

# LoRA 配置
USE_LORA=false
# USE_LORA=true 
LORA_WEIGHTS="/SafeMERGE/models/merged_models/lora_safe_instruct"
# =================================================


# --- 脚本主体 ---

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python 脚本 '$PYTHON_SCRIPT' 未找到！"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 '$INPUT_DIR' 未找到！"
    exit 1
fi

if [ "$USE_LORA" = true ]; then
    echo "模式: 基础模型 + LoRA 模块"
    echo "LoRA 路径: $LORA_WEIGHTS"
else
    echo "模式: 仅使用基础模型"
fi
echo "将使用 ${#GPUS[@]} 个GPU: ${GPUS[*]}"
echo "每张GPU的最大并行作业数限制为: $MAX_MODEL_PER_CUDA"
echo ""

# --- 任务定义 ---

# 将所有待处理的json文件存入一个数组
files_to_process=("$INPUT_DIR"/*.json)

# 创建所有任务 (model, file) 的组合
declare -a all_tasks_model
declare -a all_tasks_file

echo "正在生成任务列表..."
for model in "${MODEL_LIST[@]}"; do
    for file in "${files_to_process[@]}"; do
        all_tasks_model+=("$model")
        all_tasks_file+=("$file")
    done
done

total_tasks=${#all_tasks_model[@]}
if [ $total_tasks -eq 0 ]; then
    echo "错误：在 $INPUT_DIR 中未找到 .json 文件或 MODEL_LIST 为空。"
    exit 1
fi
echo "总共需要处理 $total_tasks 个任务 ( ${#MODEL_LIST[@]} 个模型 x ${#files_to_process[@]} 个文件 )。"


# --- 并行处理逻辑 ---

# 跟踪每个GPU上的活动作业数
declare -A gpu_job_count
for gpu in "${GPUS[@]}"; do
    gpu_job_count[$gpu]=0
done

# 跟踪 PID 及其分配的 GPU
declare -A pid_to_gpu

tasks_launched=0
tasks_completed=0

# 定义一个函数来启动任务
# 注意：此函数现在需要 MODEL_NAME 作为参数
run_inference() {
    local model_name=$1
    local file_path=$2
    local gpu_id=$3
    
    # --- 根据 model_name 动态配置路径 ---
    local output_dir="/safety-tuned-llamas/safety_eval/$MODEL_CLASS/$model_name"
    local base_model="/SafeMerge/merged_models/$MODEL_CLASS/$model_name"
    
    # 创建输出目录（如果不存在）
    mkdir -p "$output_dir"
    
    local filename=$(basename "$file_path")
    local output_file="$output_dir/$filename"

    echo "[GPU-$gpu_id] [${model_name}] 开始处理文件: $filename"
    
    # 使用数组构建命令行参数
    local cmd_args=(
        "--base_model" "$base_model"
        "--input_path" "$file_path"
        "--output_path" "$output_file"
        "--prompt_template_path" "$PROMPT_TEMPLATE"
        "--gpu_id" "$gpu_id"
    )

    # 如果 USE_LORA 为 true 并且 LORA_WEIGHTS 变量不为空，则添加 lora_weights 参数
    if [ "$USE_LORA" = true ] && [ -n "$LORA_WEIGHTS" ]; then
        cmd_args+=("--lora_weights" "$LORA_WEIGHTS")
    fi
    
    # 在后台执行 Python 脚本，并将标准输出和错误重定向到日志文件
    python "$PYTHON_SCRIPT" "${cmd_args[@]}" > "${output_file}.log" 2>&1 &
}


# --- 主循环 (已修改以兼容旧版Bash) ---
echo "开始处理任务..."

while [ $tasks_completed -lt $total_tasks ]; do
    
    # 标志：是否在这一轮循环中启动了新任务
    launched_new_job=false
    
    # 1. 尝试启动新任务 (如果还有任务且GPU有空闲)
    if [ $tasks_launched -lt $total_tasks ]; then
        # 遍历可用的GPU
        for gpu_id in "${GPUS[@]}"; do
            # 检查此GPU是否已满
            while [ ${gpu_job_count[$gpu_id]} -lt $MAX_MODEL_PER_CUDA ] && [ $tasks_launched -lt $total_tasks ]; do
                # 此GPU有空位，启动一个新任务
                
                # 获取下一个任务
                model_to_run="${all_tasks_model[$tasks_launched]}"
                file_to_run="${all_tasks_file[$tasks_launched]}"
                
                # 启动任务
                run_inference "$model_to_run" "$file_to_run" "$gpu_id"
                new_pid=$!
                
                # 记录这个新任务
                pid_to_gpu[$new_pid]=$gpu_id
                gpu_job_count[$gpu_id]=$(( ${gpu_job_count[$gpu_id]} + 1 ))
                tasks_launched=$((tasks_launched + 1))
                launched_new_job=true
                
                echo "已启动任务 $tasks_launched / $total_tasks (PID: $new_pid, GPU: $gpu_id)。GPU $gpu_id 当前负载: ${gpu_job_count[$gpu_id]} / $MAX_MODEL_PER_CUDA"

                # 如果所有任务都已启动，跳出内部的 while 循环
                if [ $tasks_launched -eq $total_tasks ]; then
                    echo "所有 $total_tasks 个任务均已启动，等待完成..."
                    break
                fi
            done
            
            # 如果所有任务都启动了，也跳出外部的 for 循环
            if [ $tasks_launched -eq $total_tasks ]; then
                break
            fi
        done
    fi

    # 2. 检查已完成的任务 (替代 'wait -n -p')
    # 如果没有启动新任务 (因为GPU已满或所有任务已启动)
    # 并且仍然有任务在运行，我们需要等待
    
    current_jobs_count=${#pid_to_gpu[@]}
    
    if [ "$launched_new_job" = false ] && [ $current_jobs_count -gt 0 ]; then
        
        found_completed_job=false
        while [ "$found_completed_job" = false ]; do
            # 遍历所有正在跟踪的PID
            for pid in "${!pid_to_gpu[@]}"; do
                # 检查进程是否仍然存在
                # kill -0 $pid 会在进程存在时返回true, 进程不存在时返回false
                if ! kill -0 "$pid" 2>/dev/null; then
                    # 进程已结束
                    
                    # 使用 wait $pid 来回收进程并获取其退出状态
                    if ! wait "$pid"; then
                        echo "一个后台任务 (PID: $pid) 执行失败，请检查日志。"
                    fi
                    
                    tasks_completed=$((tasks_completed + 1))
                    
                    # 查找它在哪个GPU上
                    gpu_id=${pid_to_gpu[$pid]}
                    
                    # 减少该GPU的计数
                    gpu_job_count[$gpu_id]=$(( ${gpu_job_count[$gpu_id]} - 1 ))
                    
                    echo "任务 (PID: $pid) 已完成。GPU $gpu_id 剩余 ${gpu_job_count[$gpu_id]} 个作业。总计 $tasks_completed / $total_tasks 完成。"

                    # 从PID跟踪中移除
                    unset pid_to_gpu[$pid]
                    
                    # 标记我们找到了一个已完成的作业，可以退出轮询循环
                    found_completed_job=true
                    
                    # 退出 'for pid' 循环
                    break
                fi
            done # 结束 for pid 循环
            
            # 如果在这一轮检查中没有找到已完成的作业，
            # 并且仍有作业在运行，则短暂休眠
            if [ "$found_completed_job" = false ]; then
                # 确保我们没有在所有作业都完成后陷入死循环
                if [ ${#pid_to_gpu[@]} -gt 0 ]; then
                    sleep 1
                else
                    # 似乎所有作业都消失了，但没有被正确捕获
                    break
                fi
            fi
            
        done # 结束 while found_completed_job 循环
    fi # 结束 if launched_new_job=false
    
    # 安全退出：如果所有任务都已完成
    if [ $tasks_completed -ge $total_tasks ]; then
        break
    fi

    # 安全退出：如果任务尚未全部完成，但没有作业在运行了
    if [ $tasks_launched -eq $total_tasks ] && [ ${#pid_to_gpu[@]} -eq 0 ]; then
         if [ $tasks_completed -lt $total_tasks ]; then
            echo "警告：所有子进程均已退出，但完成的任务数 ($tasks_completed) 少于总任务数 ($total_tasks)。"
         fi
         break
    fi
    
done # 结束主 while 循环

echo ""
echo "所有 $total_tasks 个任务均已处理完毕！"