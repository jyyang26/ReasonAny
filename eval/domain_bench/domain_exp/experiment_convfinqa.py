import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
import re
from pathlib import Path

def main():
    # --- 1. 参数解析 ---
    parser = argparse.ArgumentParser(description="Run ConvFinQA evaluation.")
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--device_number', type=int, default=0, help='GPU device ID')
    parser.add_argument('--model_path', type=str, required=True, help='Full path to the model')
    parser.add_argument('--save_path', type=str, required=True, help='Full path to save results.json')
    parser.add_argument('--data_path', type=str, 
                        default="ConvFinQA/test.json",
                        help='Path to the input dataset')
    parser.add_argument('--reasoning', action='store_true', help='Use reasoning generation parameters')
    
    args = parser.parse_args()

    # --- 2. 变量配置 ---
    device_name = args.device_number
    model_name = args.model_name
    model_path = args.model_path
    input_json_path = args.data_path
    save_path = Path(args.save_path)
    reasoning = args.reasoning

    # 确保保存目录存在 (按照要求: .../{model_name}/results.json)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # System prompt template (保留原逻辑)
    system_prompt_template = """You are an economics-focused result generation machine. Your task is to analyze the provided "input" text and accurately answer the question presented at the end of the text. The answer should always be a floating-point number.

Input: {input_text}
Output: """

    # --- 3. 加载资源 ---
    device = torch.device(f"cuda:{device_name}" if torch.cuda.is_available() else "cpu")
    print(f"[{model_name}] INFO: Using device: {device}")

    print(f"[{model_name}] INFO: Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"[{model_name}] ERROR: Failed to load tokenizer: {e}")
        return

    print(f"[{model_name}] INFO: Loading model from {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            device_map=f"cuda:{device_name}" # 强制指定到具体显卡
        )
        model.eval()
    except Exception as e:
        print(f"[{model_name}] ERROR: Failed to load model: {e}")
        return

    # 处理 Pad Token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
        else:
            print(f"[{model_name}] INFO: Please ensure the model has a pad_token_id or eos_token_id.")

    # --- 4. 加载数据 ---
    print(f"[{model_name}] INFO: Loading data from {input_json_path}...")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"[{model_name}] ERROR: Could not load data: {e}")
        return

    results = []
    print(f"[{model_name}] INFO: Processing {len(test_data)} items...")

    # --- 5. 推理循环 ---
    for i, item in enumerate(test_data):
        input_text = item.get("input")

        if not isinstance(input_text, str):
            results.append({**item, "generated_output_raw": None, "generated_output_float": None, "error": "Invalid input"})
            continue

        prompt = system_prompt_template.format(input_text=input_text)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings - (32768 if reasoning else 2048)).to(device)
        input_ids_length = inputs.input_ids.shape[1]

        try:
            with torch.no_grad():
                if reasoning:
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=32768,
                        temperature=0.6,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                else:
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=4096, # Changed from max_length to max_new_tokens for safety
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )

            generated_ids = outputs[0][input_ids_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # 解析浮点数 (保留原逻辑)
            parsed_float = None
            if generated_text:
                try:
                    match = re.search(r"[-+]?\d*\.\d+|\d+", generated_text)
                    if match:
                        parsed_float = float(match.group(0))
                except ValueError:
                    pass

            current_result = {**item}
            current_result["generated_output_raw"] = generated_text
            current_result["generated_output_float"] = parsed_float
            results.append(current_result)
            
            if i % 50 == 0 or i == 0:
                print(f"[{model_name}] 进度: {i+1}/{len(test_data)} | 结果: {parsed_float}")

        except Exception as e:
            print(f"[{model_name}] ERROR Item {i}: {e}")
            results.append({**item, "error": str(e)})
            
        # 简单的显存清理
        if i % 10 == 0:
            torch.cuda.empty_cache()

    # --- 6. 保存结果 ---
    print(f"[{model_name}] INFO: Saving results to {save_path}...")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"--- 成功完成: {model_name} ---")

if __name__ == "__main__":
    main()