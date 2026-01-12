# inference.py (修改后)

import os
import sys
import json
import os.path as osp
from typing import Union

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

from tap import Tap

# Check if GPU is available
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Check if MPS is available
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


# Arguments class
class Arguments(Tap):
    ## Model parameters
    base_model: str = ""
    # ==================== 代码修改处 START: 将 lora_weights 设为可选 ====================
    # 将 lora_weights 的默认值设置为空字符串，这样当不提供此参数时，我们知道不加载LoRA
    lora_weights: str = ""
    # ==================== 代码修改处 END ============================================
    load_8bit: bool = False
    auth_token: str = ""

    ## Generation parameters
    max_new_tokens: int = 256
    num_beams: int = 4
    top_k: int = 40
    top_p: float = 0.75
    temperature: float = 0.1

    ## Input and output files
    prompt_template_path: str = "../../configs/alpaca.json"
    input_path: str = "data/test_input.json"
    # The full path for the output JSON file
    output_path: str = "../output/test_output.json"
    
    # 添加一个新的参数来指定要使用的 GPU 索引
    gpu_id: int = 0


# Prompter class
class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = template_name  # osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


# Evaluation function
def evaluate(
    model,
    tokenizer,
    prompter,
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    stream_output=False,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    # 注意: 模型加载时已通过 device_map 指定了设备，这里的 .to(device) 实际上是 to('cuda')
    # 对于多卡并行，这没有问题，因为每个进程只会看到 CUDA_VISIBLE_DEVICES 指定的卡
    # 或者模型内部会根据 device_map 正确处理张量位置
    input_ids = inputs["input_ids"].to(f"cuda:{model.device.index}")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return prompter.get_response(output)


# Main function
def main(args: Arguments):
    # Load the input data (.json)
    input_path = args.input_path
    with open(input_path) as f:
        input_data = json.load(f)
    instructions = input_data["instructions"]
    inputs = input_data["inputs"]

    # Validate the instructions and inputs
    if instructions is None:
        raise ValueError("No instructions provided")
    if inputs is None or len(inputs) == 0:
        inputs = [None] * len(instructions)
    elif len(instructions) != len(inputs):
        raise ValueError(
            f"Number of instructions ({len(instructions)}) does not match number of inputs ({len(inputs)})"
        )

    # Load the prompt template
    prompter = Prompter(args.prompt_template_path)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # ==================== 代码修改处 START: 根据 lora_weights 条件加载模型 ====================
    # 删除了 load_in_8bit 参数，模型将以半精度 (FP16) 加载。
    if device == "cuda":
        # 使用从命令行传入的 gpu_id
        target_device = f"cuda:{args.gpu_id}"
        print(f"Loading base model on device: {target_device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map=target_device, # 使用动态设备
            trust_remote_code=True,
        )
        # 如果提供了 lora_weights 路径，则加载 LoRA 模块
        if args.lora_weights:
            print(f"Loading LoRA weights from: {args.lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                torch_dtype=torch.float16,
            )
        else:
            print("No LoRA weights provided, using base model only.")
    
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        # 如果提供了 lora_weights 路径，则加载 LoRA 模块
        if args.lora_weights:
            print(f"Loading LoRA weights from: {args.lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            print("No LoRA weights provided, using base model only.")
    else: # cpu
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        # 如果提供了 lora_weights 路径，则加载 LoRA 模块
        if args.lora_weights:
            print(f"Loading LoRA weights from: {args.lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                device_map={"": device},
            )
        else:
            print("No LoRA weights provided, using base model only.")
    # ==================== 代码修改处 END ======================================

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        # 在多进程并行时，torch.compile 可能不是最佳选择，可以考虑注释掉
        # model = torch.compile(model)
        pass


    # Generate the outputs
    outputs = []
    # 在 tqdm 的描述信息中加入 GPU ID，方便监控
    desc = f"GPU-{args.gpu_id}: Evaluating {os.path.basename(input_path)}"
    for instruction, input in tqdm(
        zip(instructions, inputs),
        total=len(instructions),
        desc=desc,
    ):
        output = evaluate(
            model=model,
            tokenizer=tokenizer,
            prompter=prompter,
            instruction=instruction,
        )
        outputs.append(output)

    # Save the outputs
    output_path = args.output_path

    # Check if the output path directory exists
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    # Save the outputs to the output path
    print(f"GPU-{args.gpu_id}: Saving results to {output_path}...")
    with open(output_path, "w") as f:
        # 同样，从保存的参数中移除 load_8bit
        json.dump(
            {
                "parameters": {
                    "model": args.base_model,
                    "prompt_template": args.prompt_template_path,
                    "lora_weights": args.lora_weights or "None", # 如果为空则记录为'None'
                },
                "inputs": inputs,
                "instructions": instructions,
                "outputs": outputs,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)