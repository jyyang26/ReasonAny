import json
import numpy as np
import argparse
from evals.HarmfulnessRewardModel import AbsoluteHarmfulnessPredictor, ConversationBuilder
import os

def process_json_file(file_path, device):
    """
    对单个JSON文件计算safety_value并更新文件。
    
    Args:
        file_path (str): JSON文件的完整路径。
        device (str): 使用的计算设备 (例如, 'cuda:0')。
    """
    print(f"[{device}] 开始处理文件: {os.path.basename(file_path)}")

    # 1. 读取JSON文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[{device}] 错误：找不到文件 {file_path}")
        return
    except json.JSONDecodeError:
        print(f"[{device}] 错误：文件 {file_path} 不是一个有效的JSON文件。")
        return

    # 2. 从JSON对象中提取user_texts和assistant_texts
    if 'instructions' not in data or 'outputs' not in data:
        print(f"[{device}] 错误：JSON文件 {os.path.basename(file_path)} 中必须包含 'instructions' 和 'outputs' 键。")
        return

    user_texts = data['instructions']
    assistant_texts = data['outputs']

    # 3. 初始化模型并进行预测
    setup = "redteam"
    harmfulness_predictor = AbsoluteHarmfulnessPredictor(setup, device=device)
    harmfulness_scores = harmfulness_predictor.predict(user_texts, assistant_texts)

    # 4. 将预测结果（安全评分）添加回JSON对象
    # 如果harmfulness_scores是numpy数组，需要转换；如果已经是list，则直接赋值
    if hasattr(harmfulness_scores, 'tolist'):
        data['safety_value'] = harmfulness_scores.tolist()
    else:
        data['safety_value'] = harmfulness_scores # 已经是list了

    # 5. 将更新后的内容写回同一个JSON文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[{device}] 写入文件 {os.path.basename(file_path)} 时发生错误：{e}")
        return

    # 6. 计算并输出safety_value的平均值
    if len(harmfulness_scores) > 0:
        average_score = np.mean(harmfulness_scores)
        print(f"[{device}] ✔️ 处理完成: {os.path.basename(file_path)} | 平均分: {average_score:.4f}")
    else:
        print(f"[{device}] 警告: {os.path.basename(file_path)} 没有可计算平均值的safety_value。")


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="为JSON文件计算并添加Safety Value")
    parser.add_argument('--file_path', type=str, required=True, help='需要处理的JSON文件的路径')
    parser.add_argument('--device', type=str, required=True, help="计算设备，例如 'cuda:0'")
    
    args = parser.parse_args()
    
    process_json_file(args.file_path, args.device)