#!/bin/bash

# =======================
# Configuration for GPU 0
# =======================
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_MULTIPROCESSING_SPAWN=1
export PYTHON_MULTIPROCESSING_METHOD=spawn

# Common Settings
temperature=0.0
method=DirectRequest
behaviors_path=./data/behavior_datasets/harmbench_behaviors_text_all.csv
base_save_dir=./results_harmbench_temp${temperature}
num_tokens=4096
cls_path=models/HarmBench-Llama-2-13b-cls

# Models assigned to GPU 0
models=(
    # --- Llama 8B Group ---
    MODEL_NAMES
)

# Execution Loop
for model in "${models[@]}"; do
    echo "========================================================"
    echo "[GPU 0] Processing model: $model"
    echo "========================================================"
    
    if [ -f "./scripts/run_pipeline.py" ]; then
        python ./scripts/run_pipeline.py \
            --methods $method \
            --models $model \
            --base_save_dir $base_save_dir \
            --behaviors_path $behaviors_path \
            --step all \
            --mode local \
            --max_new_tokens $num_tokens \
            --temperature $temperature \
            --cls_path $cls_path
    else
        echo "Script ./scripts/run_pipeline.py not found, using standard breakdown..."
        python -u generate_test_cases.py \
            --method_name=$method \
            --experiment_name="default" \
            --behaviors_path=$behaviors_path \
            --save_dir="${base_save_dir}/${method}/default/test_cases" \
            --behavior_start_idx=0 --behavior_end_idx=400
        
        python -u generate_completions.py \
            --model_name=$model \
            --models_config_file=./configs/model_configs/models.yaml \
            --behaviors_path=$behaviors_path \
            --test_cases_path="${base_save_dir}/${method}/default/test_cases/test_cases.json" \
            --save_path="${base_save_dir}/${method}/default/completions/${num_tokens}_tokens/${model}.json" \
            --max_new_tokens=$num_tokens \
            --temperature=$temperature \
            --generate_with_vllm=True
            
        python -u evaluate_completions.py \
            --cls_path=$cls_path \
            --behaviors_path=$behaviors_path \
            --completions_path="${base_save_dir}/${method}/default/completions/${num_tokens}_tokens/${model}.json" \
            --save_path="${base_save_dir}/${method}/default/results/${num_tokens}_tokens/${model}.json" \
            --save_asr_path="${base_save_dir}/${method}/default/results/${num_tokens}_tokens/${model}_asr.json" \
            --include_advbench_metric=True
    fi
done