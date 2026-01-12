import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import torch
import argparse
from pathlib import Path

def main():
    # --- 1. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Run model evaluation on HumanEval.")
    parser.add_argument('--model_name', type=str, required=True, 
                        help='要评估的模型的名称 (例如 "dare_dpsk_med")')
    parser.add_argument('--device_number', type=int, required=True, 
                        help='要使用的 GPU 设备编号 (例如 0 或 1)')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型的完整路径')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='结果 JSON 文件的完整保存路径')
    parser.add_argument('--model_class', type=str, default="qwen_7b", 
                        help='模型类别 (例如 "qwen_7b" 或 "llama_7b")')
    parser.add_argument('--test_dataset_path', type=str, 
                        default="openai_humaneval/openai_humaneval/test-00000-of-00001.parquet",
                        help='HumanEval parquet 文件的路径')
    parser.add_argument('--use_auto', default=False,
                        help='使用 device_map="auto" 加载模型')
    parser.add_argument('--reasoning', action='store_true', 
                        help='使用 reasoning 模式进行生成')
    args = parser.parse_args()

    # --- 2. 使用传入的参数设置变量 ---
    model_name = args.model_name
    device_number = args.device_number
    model_path = args.model_path
    save_path = Path(args.save_path)
    model_class = args.model_class
    test_dataset_path = args.test_dataset_path
    use_reasoning = args.reasoning
    use_auto = args.use_auto

    print(f"--- 开始处理模型: {model_name} on GPU: {device_number} ---")
    print(f"模型路径: {model_path}")
    print(f"数据路径: {test_dataset_path}")
    print(f"保存路径: {save_path}")

    # 确保保存目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 3. 加载数据集 ---
    print(f"[{model_name}] 正在加载数据集...")
    df = pd.read_parquet(test_dataset_path)
    
    # --- 4. 加载模型和分词器 ---
    device_str = f"cuda:{device_number}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if use_auto:
        print(f"[{model_name}] 使用 device_map='auto' 加载模型")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        print(f"[{model_name}] 正在加载模型到: {device_str}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_str
        )
    
    # 您的原始代码使用了 local_files_only=True，如果不确定是否需要，可以移除它
    # tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    def generate_answer(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        gen_kwargs = {
            "pad_token_id": tokenizer.eos_token_id
        }
        if use_reasoning:
            gen_kwargs.update({
                "max_new_tokens": 32768,
                "temperature": 0.6,
                "top_p": 0.95,
                "do_sample": True
            })
        else:
            gen_kwargs.update({
                "max_new_tokens": 4096,
            })

        outputs = model.generate(
            inputs.input_ids,
            **gen_kwargs
        )
        return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    # --- 5. 处理数据集和推理 ---
    results = []
    print(f"[{model_name}] 开始评估...")
    for idx, row in df.iterrows():
        prompt_text = row['prompt']
        
        # 根据 model_class 动态构建提示
        if "llama" in model_class.lower():
            # 为 Llama 构建特定格式的提示
            system_prompt = """You are an expert Python programmer. Your task is to complete the Python function based on the provided problem description and function signature.

Only output the raw code for the function's body. Do not include the function signature, the docstring, or any other explanatory text."""
            full_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt_text} [/INST]"""
        else:
            # 为 Qwen 或其他模型构建通用提示 (基于您原始脚本的 'else' 逻辑)
            system_prompt = "You are an expert Python programmer. Complete the function below according to the problem description."
            # 注意：非 Llama 模型不应包含 <s>[INST] 标签
            full_prompt = f"{system_prompt}\n\n{prompt_text}"

        
        generated_text = generate_answer(full_prompt)
        
        # 收集结果
        result = {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
            "answer": generated_text  # 添加生成的答案
        }
        
        if idx % 20 == 0 or idx == 0: # 每 20 个打印一次进度
             print(f"[{model_name} @ GPU{device_number}] 进度: {idx+1}/{len(df)}")
             print(f"生成预览: {generated_text[:80]}...")
             
        results.append(result)

    # --- 6. 保存结果 ---
    print(f"[{model_name}] 评估完成。正在保存 {len(results)} 条结果到 {save_path}...")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"--- 成功完成: {model_name} on GPU: {device_number} ---")


if __name__ == "__main__":
    main()