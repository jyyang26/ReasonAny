import argparse
import os
import gc
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 核心算法實現 ---

def get_task_vector(base_model, finetuned_model):
    """計算任務向量 (finetuned_model - base_model)。"""
    task_vector = {}
    base_params = dict(base_model.named_parameters())
    finetuned_params = dict(finetuned_model.named_parameters())

    with torch.no_grad():
        for name in base_params:
            if name in finetuned_params:
                task_vector[name] = (finetuned_params[name].data - base_params[name].data).cpu()
    return task_vector

def calculate_importance_scores(model, tokenizer, texts, device, is_gradient_base=True):
    """
    計算模型中每個權重的重要性分數。
    - is_gradient_base=True:  使用梯度幅值 |g|
    - is_gradient_base=False: 使用 SNIP 分數 |w*g|
    """
    model.to(device)
    model.eval()

    importance_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    
    score_type = "Gradient" if is_gradient_base else "SNIP"
    print(f"Calculating {score_type} scores over {len(texts)} samples...")

    for text in tqdm(texts, desc=f"{score_type} calculation"):
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)
        labels = inputs["input_ids"]

        model.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if is_gradient_base:
                        importance_scores[name] += torch.abs(param.grad)
                    else:
                        importance_scores[name] += torch.abs(param.data * param.grad)
    
    for name in importance_scores:
        importance_scores[name] /= len(texts)
        
    return importance_scores

def get_selected_param_names(scores, ratio, select_lowest=False):
    """根據分數和比例，選取參數名稱 (最高或最低)。"""
    all_scores = torch.cat([v.flatten() for v in scores.values()])
    k = int(len(all_scores) * ratio)
    
    if not k > 0:
        print(f"Warning: Calculated k=0 for ratio {ratio}. No parameters will be selected.")
        return set()

    if select_lowest:
        threshold = torch.kthvalue(all_scores, k).values
        print(f"Selecting BOTTOM {ratio*100:.2f}% of params. Threshold: {threshold.item()}")
    else:
        threshold = torch.kthvalue(all_scores, len(all_scores) - k).values
        print(f"Selecting TOP {ratio*100:.2f}% of params. Threshold: {threshold.item()}")

    selected_names = set()
    for name, score_tensor in scores.items():
        if select_lowest:
            if (score_tensor <= threshold).any():
                 selected_names.add(name)
        else:
            if (score_tensor >= threshold).any():
                selected_names.add(name)
    
    return selected_names


def load_texts(file_path, file_type, num_samples, text_field_logic):
    """從 Parquet 文件中加載文本數據。"""
    print(f"Loading {num_samples} samples from {file_path}...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[ERROR] Could not load or parse Parquet file: {file_path}\n{e}")
        return []
        
    df = df.head(num_samples)
    texts = []
    
    for _, row in df.iterrows():
        try:
            text = text_field_logic(row)
            if text:
                texts.append(text)
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Skipping a row due to format error: {e}")
            continue

    print(f"Successfully loaded {len(texts)} texts for '{file_type}'.")
    return texts

# --- 主函數 ---

def main():
    parser = argparse.ArgumentParser(description="Ablation Study script.")
    # (參數定義部分)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--safety_model_path", type=str, required=True)
    parser.add_argument("--reasoning_model_path", type=str, required=True)
    parser.add_argument("--safety_dataset_path", type=str, required=True)
    parser.add_argument("--reasoning_dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--safety_ratio", type=float, default=0.1)
    parser.add_argument("--reasoning_ratio", type=float, default=0.1)
    parser.add_argument("--safety_scale", type=float, default=1.0)
    parser.add_argument("--reasoning_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Ablation Flags
    parser.add_argument("--is_intersect", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--is_reasoning_top", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--is_gradient_base", type=lambda x: (str(x).lower() == 'true'), default=True)
    
    # **【新增】Ablation Control Flags**
    parser.add_argument("--is_use_safety", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to include safety model merging")
    parser.add_argument("--is_use_reasoning", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to include reasoning model merging")

    args = parser.parse_args()

    # ==================== 1. 加載 Base Model & Tokenizer ====================
    print("--- 1. Loading Base Model & Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
    base_model.to(args.device)

    # 初始化空的集合和向量，用於處理消融情況
    T_safety = set()
    T_reasoning = set()
    safety_task_vector = {}
    reasoning_task_vector = {}

    # ==================== 2. 安全模型處理 (Ablation Control) ====================
    if args.is_use_safety:
        print("\n--- [Safety Module] ENABLED ---")
        # 只在需要時加載數據
        safety_texts = load_texts(args.safety_dataset_path, "safety", args.num_samples, lambda row: row['chosen'])
        
        if safety_texts:
            print("Loading safety model...")
            safety_model = AutoModelForCausalLM.from_pretrained(args.safety_model_path, torch_dtype=torch.bfloat16)
            safety_model.to(args.device)

            print("Calculating safety task vector...")
            safety_task_vector = get_task_vector(base_model, safety_model)
            
            print("Calculating importance scores for safety...")
            safety_model_scores = calculate_importance_scores(safety_model, tokenizer, safety_texts, args.device, args.is_gradient_base)
            
            if args.is_intersect:
                # Intersect Logic
                base_on_safety_scores = calculate_importance_scores(base_model, tokenizer, safety_texts, args.device, args.is_gradient_base)
                N_safety = get_selected_param_names(safety_model_scores, args.safety_ratio, select_lowest=False)
                N_base_for_safety = get_selected_param_names(base_on_safety_scores, args.safety_ratio, select_lowest=False)
                T_safety = N_safety.intersection(N_base_for_safety)
                del base_on_safety_scores, N_base_for_safety
            else:
                N_safety = get_selected_param_names(safety_model_scores, args.safety_ratio, select_lowest=False)
                T_safety = N_safety
            
            print(f"Elected {len(T_safety)} params for Safety Task.")
            
            # Cleanup
            print("Releasing safety model and scores...")
            del safety_model, safety_model_scores, N_safety
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("[Warning] Safety texts loading failed, skipping Safety module.")
    else:
        print("\n--- [Safety Module] DISABLED (Ablation) ---")

    # ==================== 3. 推理模型處理 (Ablation Control) ====================
    if args.is_use_reasoning:
        print("\n--- [Reasoning Module] ENABLED ---")
        # 只在需要時加載數據
        reasoning_texts = load_texts(args.reasoning_dataset_path, "reasoning", args.num_samples, lambda row: row['conversations'][0]['value'] + tokenizer.eos_token + row['conversations'][1]['value'])

        if reasoning_texts:
            print("Loading reasoning model...")
            reasoning_model = AutoModelForCausalLM.from_pretrained(args.reasoning_model_path, torch_dtype=torch.bfloat16)
            reasoning_model.to(args.device)

            print("Calculating reasoning task vector...")
            reasoning_task_vector = get_task_vector(base_model, reasoning_model)

            print("Calculating importance scores for reasoning...")
            reasoning_model_scores = calculate_importance_scores(reasoning_model, tokenizer, reasoning_texts, args.device, args.is_gradient_base)
            
            select_lowest_for_reasoning = not args.is_reasoning_top
            strategy_str = "LOWEST" if select_lowest_for_reasoning else "HIGHEST"
            print(f"Strategy: Selecting {strategy_str} scores for reasoning model.")
            
            if args.is_intersect:
                base_on_reasoning_scores = calculate_importance_scores(base_model, tokenizer, reasoning_texts, args.device, args.is_gradient_base)
                N_reasoning = get_selected_param_names(reasoning_model_scores, args.reasoning_ratio, select_lowest=select_lowest_for_reasoning)
                N_base_for_reasoning = get_selected_param_names(base_on_reasoning_scores, args.reasoning_ratio, select_lowest=select_lowest_for_reasoning)
                T_reasoning = N_reasoning.intersection(N_base_for_reasoning)
                del base_on_reasoning_scores, N_base_for_reasoning
            else:
                N_reasoning = get_selected_param_names(reasoning_model_scores, args.reasoning_ratio, select_lowest=select_lowest_for_reasoning)
                T_reasoning = N_reasoning # 修正：如果不 Intersect，直接選 Reasoning 自己的參數

            print(f"Elected {len(T_reasoning)} params for Reasoning Task.")
            
            # Cleanup
            print("Releasing reasoning model and scores...")
            del reasoning_model, reasoning_model_scores, N_reasoning
            gc.collect()
            torch.cuda.empty_cache()
        else:
             print("[Warning] Reasoning texts loading failed, skipping Reasoning module.")
    else:
        print("\n--- [Reasoning Module] DISABLED (Ablation) ---")

    # ==================== 4. 合併與保存 ====================
    print("\n--- 4. Disjoint Step ---")
    # 如果某一項為空集（被禁用），集合減法仍能正常工作：A - empty = A; empty - A = empty
    T_safety_disjoint = T_safety - T_reasoning
    T_reasoning_disjoint = T_reasoning - T_safety
    
    print(f"Disjoint Safety Params to apply: {len(T_safety_disjoint)}")
    print(f"Disjoint Reasoning Params to apply: {len(T_reasoning_disjoint)}")

    print("\n--- 5. Merging ---")
    base_model.to("cpu")
    merged_state_dict = base_model.state_dict()
    
    with torch.no_grad():
        # 只有在啟用了 Safety 且計算了 vector 的情況下才執行
        if args.is_use_safety and safety_task_vector:
            for name in tqdm(T_safety_disjoint, desc="Applying safety vector"):
                if name in merged_state_dict:
                    merged_state_dict[name] += args.safety_scale * safety_task_vector[name]
        
        # 只有在啟用了 Reasoning 且計算了 vector 的情況下才執行
        if args.is_use_reasoning and reasoning_task_vector:
            for name in tqdm(T_reasoning_disjoint, desc="Applying reasoning vector"):
                if name in merged_state_dict:
                    merged_state_dict[name] += args.reasoning_scale * reasoning_task_vector[name]

    print("\n--- 6. Saving Merged Model ---")
    final_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
    final_model.load_state_dict(merged_state_dict)
    
    os.makedirs(args.output_path, exist_ok=True)
    final_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"Merged model saved to {args.output_path}")

if __name__ == "__main__":
    main()