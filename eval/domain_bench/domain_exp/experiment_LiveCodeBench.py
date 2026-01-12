import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import argparse
from pathlib import Path

def main():
    # --- 1. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Run model evaluation on LiveCodeBench.")
    parser.add_argument('--model_name', type=str, required=True, 
                        help='要评估的模型的名称')
    parser.add_argument('--device_number', type=int, required=True, 
                        help='要使用的 GPU 设备编号 (例如 0 或 1)')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型的完整路径')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='结果 JSON 文件的完整保存路径')
    parser.add_argument('--data_path', type=str, 
                        default="code_generation/test.jsonl",
                        help='LiveCodeBench test.jsonl 文件的路径')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='推理时使用的批量大小')
    parser.add_argument('--use_auto', action='store_true', 
                        help='使用 device_map="auto" 加载模型')
    parser.add_argument('--reasoning', action='store_true', 
                        help='使用 reasoning 模式进行生成')
    args = parser.parse_args()

    # --- 2. 使用传入的参数设置变量 ---
    model_name = args.model_name
    device_number = args.device_number
    model_path = args.model_path
    save_path = Path(args.save_path)
    data_path = args.data_path
    use_reasoning = args.reasoning
    use_auto = args.use_auto
    batch_size = args.batch_size
    
    print(f"--- 开始处理模型: {model_name} on GPU: {device_number} ---")
    print(f"模型路径: {model_path}")
    print(f"数据路径: {data_path}")
    print(f"保存路径: {save_path}")
    print(f"批量大小: {batch_size}")

    # 确保保存目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 3. 加载数据集 (JSONL) ---
    print(f"[{model_name}] 正在加载数据集...")
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    # --- 4. 设置 Pipeline 以实现 Accelerate/Batching ---
    print(f"[{model_name}] 正在加载模型并设置 pipeline...")
    device_str = f"cuda:{device_number}" if torch.cuda.is_available() else "cpu"
    
    pipeline_args = {
        "model": model_path,
        "torch_dtype": torch.bfloat16
    }
    
    if use_auto:
        print(f"[{model_name}] 使用 device_map='auto' (Accelerate)")
        pipeline_args["device_map"] = "auto"
    else:
        print(f"[{model_name}] 正在固定到设备: {device_str}")
        pipeline_args["device"] = device_str

    # 初始化 pipeline
    pipe = pipeline("text-generation", **pipeline_args)
    
    # 修复 tokenizer pad_token_id (如果未设置)
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

    # --- 5. 准备所有提示 (Prompts) ---
    base_prompt = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the Python program\n\n"
    
    all_prompts = []
    for item in samples:
        question_content = item.get('question_content', '')
        prompt = base_prompt + f"Question:\n{question_content}\n\n"
        all_prompts.append(prompt)

    # --- 6. 批量推理 (Batch Inference) ---
    print(f"[{model_name}] 开始批量推理 {len(all_prompts)} 条数据...")
    
    # 设置生成参数
    gen_kwargs = {
        "pad_token_id": pipe.tokenizer.eos_token_id
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
            "do_sample": False
        })

    # 运行 pipeline
    # return_full_text=False 确保我们只得到新生成的文本
    outputs = pipe(
        all_prompts, 
        batch_size=batch_size, 
        return_full_text=False, 
        **gen_kwargs
    )
    print(f"[{model_name}] 推理完成。")

    # --- 7. 收集和处理结果 ---
    results = []
    for i, item in enumerate(samples):
        # outputs 结构是 [[{'generated_text': '...'}]]
        generated_text = outputs[i][0]['generated_text']
        program = generated_text.strip()

        # 收集结果
        result_entry = {
            'question_title': item.get('question_title'),
            'question_id': item.get('question_id'),
            'public_test_cases': item.get('public_test_cases'),
            'generated_program': program
        }
        results.append(result_entry)
        
        if i % 10 == 0 or i == 0:
            print(f"[{model_name} @ GPU{device_number}] 正在处理结果: {i+1}/{len(samples)}")

    # --- 8. 保存结果 ---
    print(f"[{model_name}] 评估完成。正在保存 {len(results)} 条结果到 {save_path}...")
    with open(save_path, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"--- 成功完成: {model_name} on GPU: {device_number} ---")

if __name__ == "__main__":
    main()