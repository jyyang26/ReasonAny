import os
import json
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse # 导入 argparse
from pathlib import Path

def main():
    # --- 1. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Run model evaluation on MedQA.")
    parser.add_argument('--model_name', type=str, required=True, 
                        help='要评估的模型的名称 (例如 "dare_dpsk_med")')
    parser.add_argument('--device_number', type=int, required=True, 
                        help='要使用的 GPU 设备编号 (例如 0 或 1)')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型的完整路径')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='结果 JSON 文件的完整保存路径')
    parser.add_argument('--data_dir', type=str, 
                        default="Medical/MedQA-USMLE-4-options-hf",
                        help='包含 test.json(l) 的输入数据集目录')
    parser.add_argument('--model_class', type=str, default="qwen_7b", 
                        help='模型类别')
    parser.add_argument('--use_auto', action='store_true', 
                        help='使用 device_map="auto" 加载模型')
    parser.add_argument('--reasoning', default=False, 
                        help='使用 reasoning 模式进行生成')
    args = parser.parse_args()

    # --- 2. 使用传入的参数设置变量 ---
    model_name = args.model_name
    device_number = args.device_number
    reasoning = args.reasoning
    data_dir = args.data_dir
    output_path = Path(args.save_path)
    model_dir = args.model_path
    model_class = args.model_class
    use_auto = args.use_auto
    
    # 查找数据集文件
    files = glob.glob(os.path.join(data_dir, "test.jsonl")) + glob.glob(os.path.join(data_dir, "test.json"))
    if not files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到 JSON(.l) 文件")
    dataset_path = files[0]  # 取第一个找到的文件

    print(f"--- 开始处理模型: {model_name} on GPU: {device_number} ---")
    print(f"模型路径: {model_dir}")
    print(f"数据路径: {dataset_path}")
    print(f"保存路径: {output_path}")

    # --- 3. 加载模型和分词器 ---
    device_str = f"cuda:{device_number}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    if use_auto:
        print(f"[{model_name}] 使用 device_map='auto' 加载模型")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        print(f"[{model_name}] 正在加载模型到: {device_str}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map=device_str # 分配到特定设备
        )
    model.eval()

    results = []
    prompt_template = '''You are a highly accurate medical question‐answering assistant.

You will receive:
  • input_text: A medical question stem describing a patient scenario or clinical problem.
  • ending0: The text of the first answer option.
  • ending1: The text of the second answer option.
  • ending2: The text of the third answer option.
  • ending3: The text of the fourth answer option.

Task:
  Select the single best option (ending0 through ending3) that correctly answers the question in input_text.

Output Requirement:
  • Respond with **only** the index of the chosen option: 0, 1, 2, or 3.
  • Do not include any additional words, punctuation, or explanation.

Question: {sent1}

ending0: {ending0}
ending1: {ending1}
ending2: {ending2}
ending3: {ending3}
'''

    # --- 4. 逐行读取并执行推理 ---
    print(f"[{model_name}] 开始评估...")
    line_count = 0
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            
            line_count += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[{model_name}] JSON 解析错误在行 {line_num}: {e}")
                continue

            sent1 = item.get("sent1", "")
            endings = [item.get(f"ending{i}", "") for i in range(4)]
            label = item.get("label", None)

            # 格式化提示
            prompt = prompt_template.format(
                sent1=sent1,
                ending0=endings[0],
                ending1=endings[1],
                ending2=endings[2],
                ending3=endings[3]
            )

            # 模型推理
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_kwargs = {
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id
                }
                if reasoning:
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
                
                output_ids = model.generate(**inputs, **gen_kwargs)

            pred = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

            # 收集结果
            results.append({
                "sent1": sent1,
                "ending0": endings[0],
                "ending1": endings[1],
                "ending2": endings[2],
                "ending3": endings[3],
                "label": label,
                "prediction": pred
            })
            
            if line_count % 100 == 0 or line_count == 1:
                print(f"[{model_name} @ GPU{device_number}] 进度: {line_count} | 预测: {pred}")


    # --- 5. 保存到 JSON 文件 ---
    print(f"[{model_name}] 评估完成。正在保存 {len(results)} 条结果到 {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)
        
    print(f"--- 成功完成: {model_name} on GPU: {device_number} ---")

if __name__ == "__main__":
    main()