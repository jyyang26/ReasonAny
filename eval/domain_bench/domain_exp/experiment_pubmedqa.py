import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from pathlib import Path
import json
import argparse # 导入 argparse

# --- 1. 设置命令行参数解析 ---
parser = argparse.ArgumentParser(description="Run model evaluation on PubMedQA.")
parser.add_argument('--model_name', type=str, required=True, 
                    help='要评估的模型的名称 (例如 "dare_dpsk_med")')
parser.add_argument('--device_number', type=int, required=True, 
                    help='要使用的 GPU 设备编号 (例如 0 或 1)')
parser.add_argument('--model_path', type=str, required=True, 
                    help='模型的完整路径')
parser.add_argument('--save_path', type=str, required=True, 
                    help='结果 JSON 文件的完整保存路径')
parser.add_argument('--data_path', type=str, 
                    default="PubMedQA/pqal_test_set.json",
                    help='输入数据集的路径')
parser.add_argument('--model_class', type=str, default="qwen_7b", 
                    help='模型类别')
parser.add_argument('--use_auto', action='store_true', 
                    help='使用 device_map="auto" 加载模型')
parser.add_argument('--reasoning', action='store_true', 
                    help='使用 reasoning 模式进行生成')
args = parser.parse_args()

# --- 2. 使用传入的参数设置变量 ---
use_auto = args.use_auto
model_class = args.model_class
model_name = args.model_name
device_number = args.device_number
reasoning = args.reasoning
MODEL_PATH = args.model_path # 从 --model_path 获取
SAVE_PATH = Path(args.save_path) # 从 --save_path 获取
DATA_PATH = args.data_path

# --- 3. 确保保存目录存在 (现在由 Bash 处理，但双重检查无害) ---
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"--- 开始处理模型: {model_name} on GPU: {device_number} ---")
print(f"模型路径: {MODEL_PATH}")
print(f"保存路径: {SAVE_PATH}")
print(f"数据路径: {DATA_PATH}")

# --- 4. 加载模型和分词器 ---
device = f"cuda:{device_number}" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if use_auto:
    print("使用 device_map='auto' 加载模型")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
else:
    print(f"正在加载模型到: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device # 使用完整的设备字符串
    )
model.eval()

def generate_answer(question, contexts):
    sys_prompt = """ [INST] <<SYS>> 
    You are the world's best diagnostic expert in the field of biology, here is a Question and the corresponding Context background. 
    Generate an answer from the model given a question and context. 
    Answer only with 'yes', 'no', or 'maybe'.
    <</SYS>>
    """
    prompt = f"System: {sys_prompt}\n\nQuestion: {question}\n\nAnswer: [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 统一管理生成参数
    gen_kwargs = {
        "pad_token_id": tokenizer.eos_token_id
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
            "temperature": 0.9,
            "do_sample": False
        })
        
    output = model.generate(
        inputs.input_ids,
        **gen_kwargs
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

# --- 5. 加载数据集 ---
print(f"[{model_name}] 正在加载数据集: {DATA_PATH}")
with open(DATA_PATH, "r") as f:
    dataset = json.load(f)
count = 0

# --- 6. 处理数据集 ---
print(f"[{model_name}] 开始评估...")
for key, entry in dataset.items():
    count +=1
    question = entry["QUESTION"]
    contexts = entry["CONTEXTS"]
    model_answer = generate_answer(question, contexts)
    
    if count % 100 == 0 or count == 1: # 每 100 个打印一次进度
        print(f"[{model_name} @ GPU{device_number}] 进度: {count}/{len(dataset)} | 答案: {model_answer}")
        
    dataset[key]["model_answer"] = model_answer

# --- 7. 保存输出 ---
print(f"[{model_name}] 评估完成。正在保存结果到 {SAVE_PATH}...")
with open(SAVE_PATH, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"--- 成功完成: {model_name} on GPU: {device_number} ---")