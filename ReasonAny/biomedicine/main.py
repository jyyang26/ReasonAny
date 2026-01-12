import argparse
import os
import gc
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 核心算法實現 (這部分函數已根據需求修改) ---

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
                    # **【核心修改】根據 is_gradient_base 決定計算方式**
                    if is_gradient_base:
                        # 直接使用梯度的絕對值
                        importance_scores[name] += torch.abs(param.grad)
                    else:
                        # 使用 SNIP 的計算方式
                        importance_scores[name] += torch.abs(param.data * param.grad)
    
    # 計算平均分數
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

# --- 主函數 (已修改) ---

def main():
    parser = argparse.ArgumentParser(description="Modified LED-Merging script with selectable strategy.")
    # (參數定義部分)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--domain_model_path", type=str, required=True)
    parser.add_argument("--reasoning_model_path", type=str, required=True)
    parser.add_argument("--domain_dataset_path", type=str, required=True)
    parser.add_argument("--reasoning_dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--domain_ratio", type=float, default=0.1)
    parser.add_argument("--reasoning_ratio", type=float, default=0.1)
    parser.add_argument("--domain_scale", type=float, default=1.0)
    parser.add_argument("--reasoning_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    
    # **【修改參數】** 解析 is_reasoning_top (選擇 top/bottom)
    parser.add_argument(
        "--is_reasoning_top",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="If 'True', select top scores for reasoning. If 'False', select lowest. Default: True"
    )
    # **【新增參數】** 解析 is_gradient_base (選擇 score 類型)
    parser.add_argument(
        "--is_gradient_base",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="If 'True', use gradient magnitude for scores. If 'False', use SNIP (|w*g|). Default: True"
    )
    
    args = parser.parse_args()

    # ==================== 1. 加載模型和數據 ====================
    print("--- 1. Loading Models & Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
    base_model.to(args.device)

    print("\n--- 2. Loading Datasets for Location Step ---")
    domain_texts = load_texts(args.domain_dataset_path, "domain", args.num_samples, lambda row: row['question'] + tokenizer.eos_token + row['long_answer'])
    reasoning_texts = load_texts(args.reasoning_dataset_path, "reasoning", args.num_samples, lambda row: row['conversations'][0]['value'] + tokenizer.eos_token + row['conversations'][1]['value'])
    if not domain_texts or not reasoning_texts:
        print("[ERROR] Failed to load texts. Aborting.")
        return

    # ==================== 2. 安全模型處理 ====================
    print("\n--- 3. Processing domain Model ---")
    print("Loading domain model...")
    domain_model = AutoModelForCausalLM.from_pretrained(args.domain_model_path, torch_dtype=torch.bfloat16)
    domain_model.to(args.device)

    print("Calculating domain task vector...")
    domain_task_vector = get_task_vector(base_model, domain_model)
    
    # **【修改】** 傳入 is_gradient_base 參數
    print("Calculating importance scores for domain...")
    domain_model_scores = calculate_importance_scores(domain_model, tokenizer, domain_texts, args.device, args.is_gradient_base)
    base_on_domain_scores = calculate_importance_scores(base_model, tokenizer, domain_texts, args.device, args.is_gradient_base)
    
    # 對於 domain 模型，始終選擇最高分 (select_lowest=False)
    N_domain = get_selected_param_names(domain_model_scores, args.domain_ratio, select_lowest=False)
    N_base_for_domain = get_selected_param_names(base_on_domain_scores, args.domain_ratio, select_lowest=False)
    T_domain = N_domain.intersection(N_base_for_domain)
    print(f"Elected {len(T_domain)} params for domain Task.")
    
    print("Releasing domain model and scores from memory...")
    del domain_model, domain_model_scores, base_on_domain_scores, N_domain, N_base_for_domain
    gc.collect()
    torch.cuda.empty_cache()

    # ==================== 3. 推理模型處理 ====================
    print("\n--- 4. Processing Reasoning Model ---")
    print("Loading reasoning model...")
    reasoning_model = AutoModelForCausalLM.from_pretrained(args.reasoning_model_path, torch_dtype=torch.bfloat16)
    reasoning_model.to(args.device)

    print("Calculating reasoning task vector...")
    reasoning_task_vector = get_task_vector(base_model, reasoning_model)

    # **【修改】** 傳入 is_gradient_base 參數
    print("Calculating importance scores for reasoning...")
    reasoning_model_scores = calculate_importance_scores(reasoning_model, tokenizer, reasoning_texts, args.device, args.is_gradient_base)
    base_on_reasoning_scores = calculate_importance_scores(base_model, tokenizer, reasoning_texts, args.device, args.is_gradient_base)
    
    # 根據 is_reasoning_top 參數決定選擇 top 還是 bottom
    select_lowest_for_reasoning = not args.is_reasoning_top
    if select_lowest_for_reasoning:
        print("Strategy: Selecting LOWEST scores for reasoning model as is_reasoning_top=False.")
    else:
        print("Strategy: Selecting HIGHEST scores for reasoning model as is_reasoning_top=True.")

    N_reasoning = get_selected_param_names(
        reasoning_model_scores, 
        args.reasoning_ratio, 
        select_lowest=select_lowest_for_reasoning
    )
    N_base_for_reasoning = get_selected_param_names(
        base_on_reasoning_scores, 
        args.reasoning_ratio, 
        select_lowest=select_lowest_for_reasoning
    )
    T_reasoning = N_reasoning.intersection(N_base_for_reasoning)
    print(f"Elected {len(T_reasoning)} params for Reasoning Task.")
    
    print("Releasing reasoning model and scores from memory...")
    del reasoning_model, reasoning_model_scores, base_on_reasoning_scores, N_reasoning, N_base_for_reasoning
    gc.collect()
    torch.cuda.empty_cache()

    # ==================== 4. 合併與保存 ====================
    print("\n--- 5. Disjoint Step ---")
    T_domain_disjoint = T_domain - T_reasoning
    T_reasoning_disjoint = T_reasoning - T_domain
    print(f"Disjoint domain Params: {len(T_domain_disjoint)}")
    print(f"Disjoint Reasoning Params: {len(T_reasoning_disjoint)}")

    print("\n--- 6. Merging ---")
    base_model.to("cpu")
    merged_state_dict = base_model.state_dict()
    
    with torch.no_grad():
        for name in tqdm(T_domain_disjoint, desc="Applying domain vector"):
            if name in merged_state_dict:
                merged_state_dict[name] += args.domain_scale * domain_task_vector[name]

        for name in tqdm(T_reasoning_disjoint, desc="Applying reasoning vector"):
            if name in merged_state_dict:
                merged_state_dict[name] += args.reasoning_scale * reasoning_task_vector[name]

    print("\n--- 7. Saving Merged Model ---")
    final_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
    final_model.load_state_dict(merged_state_dict)
    
    os.makedirs(args.output_path, exist_ok=True)
    final_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"Merged model saved to {args.output_path}")

if __name__ == "__main__":
    main()