# inference_gpu.py

import argparse
import os
import json
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def run_inference(args):
    """
    使用指定的模型和GPU对数据集的子集进行推理，并保存结果。
    """
    # 0. 检查设备是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")
    
    # 1. 加载模型和分词器
    print(f"[{args.model_path}] Loading model on device: {args.device}")
    
    # 手动加载模型并移动到指定设备
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 以获得更好的性能
    ).to(args.device)
    
    # 基于加载好的模型和分词器创建 pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=args.device,
    )

    # 2. 加载并截取数据集
    print(f"[{args.model_path}] Loading dataset from: {args.dataset_path}")
    df = pd.read_parquet(args.dataset_path)
    if args.num_samples > 0:
        print(f"[{args.model_path}] Using the first {args.num_samples} samples.")
        df = df.head(args.num_samples)
    
    instructions = df['instruction'].tolist()

    # 3. 准备 Prompt
    prompts = []
    if tokenizer.chat_template:
        for instruction in instructions:
            messages = [{"role": "user", "content": instruction}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
    else:
        prompts = instructions

    # 4. 执行批量推理
    print(f"[{args.model_path}] Starting inference with batch size: {args.batch_size}")
    
    # 设置推理超参数
    do_sample = args.temperature > 0 and args.top_p < 1.0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": do_sample,
    }
    print(f"[{args.model_path}] Generation parameters: {generation_kwargs}")

    # 推理时禁用tqdm，因为多进程并行输出会使进度条混乱
    outputs = []
    for out in pipe(prompts, batch_size=args.batch_size, return_full_text=False, **generation_kwargs):
        outputs.append(out[0]['generated_text'])

    # 5. 整合结果并保存
    df["output"] = outputs
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    results = df.to_dict(orient='records')
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"✔ [{args.model_path}] Results successfully saved to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a dataset with a given model and GPU.")
    
    # 路径参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the Parquet dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSON file.")
    
    # 数据和设备控制参数
    parser.add_argument("--device", type=str, required=True, help="CUDA device to use, e.g., 'cuda:1'.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to process from the dataset. 0 for all.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")

    # 推理超参数
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling. 0 means greedy decoding.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling.")

    args = parser.parse_args()
    run_inference(args)