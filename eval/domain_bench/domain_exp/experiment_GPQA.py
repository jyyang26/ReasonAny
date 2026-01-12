import os
import json
import argparse  # 1. 导入 argparse
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    # --- 1. 设置命令行参数解析 ---
    # 这些参数与你的 bash 脚本中 run_model 函数传递的参数相对应
    parser = argparse.ArgumentParser(description="GPQA Evaluation Script")
    
    parser.add_argument('--model_name', type=str, required=True, 
                        help='Name of the model (e.g., ties_dpsk_med)')
    parser.add_argument('--device_number', type=int, required=True, 
                        help='GPU device number to use (e.g., 3)')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Full path to the model directory')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='Full path to save the results JSON file')
    parser.add_argument('--reasoning', type=str, default='False', 
                        help='Whether to use reasoning mode (passed as "True" or "False" string)')
    parser.add_argument('--model_class', type=str, required=True, 
                        help='Model class (e.g., qwen_32b_qwq)')
    
    args = parser.parse_args()

    # --- 2. 使用解析的参数和你的特定路径 ---

    # (A) 从命令行参数获取变量
    model_dir = args.model_path
    output_path = args.save_path
    device_name = args.device_number
    
    # (B) 将 bash 传递的字符串 'True'/'False' 转换为布尔值
    use_reasoning = (args.reasoning.lower() == 'true')
    
    # (C) 使用你指定的 GPQA data_path
    data_path = "/opencompass/data/gpqa/gpqa_diamond.parquet"

    # --- 3. 打印配置信息 (可选, 便于调试) ---
    print(f"--- Starting GPQA Evaluation ---")
    print(f"Model Name: {args.model_name}")
    print(f"Model Class: {args.model_class}")
    print(f"Model Path: {model_dir}")
    print(f"Save Path: {output_path}")
    print(f"Device: cuda:{device_name}")
    print(f"Use Reasoning: {use_reasoning}")
    print(f"Data Path: {data_path}")

    # --- 4. 原始 Python 脚本的核心逻辑 ---
    
    # Load dataset
    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        # 使用从参数中获取的 device_map
        device_map={"": f"cuda:{device_name}"} 
    )
    print("Model loaded successfully.")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Inference on {args.model_name}'):
        question_content = row.get('question', '')
        prompt = "What is the correct answer to this question: "
        prompt += f"Question:{question_content}"

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        # 使用从参数中获取的 use_reasoning
        if use_reasoning:
            output_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=32768,
                temperature=0.6,
                top_p=0.90,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the generated portion after the prompt
        generated_text = generated[len(prompt):].strip()
        # print(generated_text) # 注释掉以减少日志输出

        # Collect fields
        result = {
            'question': question_content,
            'answer': row.get('solution', None),
            'generated_text': generated_text
        }
        results.append(result)

    # Save results as JSON list
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Inference complete. Results saved to {output_path}")

if __name__ == '__main__':
    main()