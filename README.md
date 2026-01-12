# ReasonAny

Official Code of "ReasonAny: Incorporating Reasoning Capability to Any Model via Simple and Effective Model Merging"

# Experiment Setup

This guide details how to replicate the environment on a new machine. The environment is designed for Large Language Model evaluation and includes libraries such as `vllm`, `deepspeed`, `transformers`, and `torch`.

## Prerequisites

* **OS:** Linux (Recommended for `vllm` and `deepspeed` compatibility)
* **Hardware:** NVIDIA GPU with appropriate drivers installed.
* **Software:** [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

## Installation Steps

### 1. Clone/Prepare Files
Ensure you have the `requirements.txt` file in your current directory.

### 2. Create the Conda Environment
Initialize a clean environment with **Python 3.10** (matching the original environment version).

```bash
conda create -n llmeval python=3.10 -y
conda activate llmeval

pip install --upgrade pip setuptools wheel
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url [https://download.pytorch.org/whl/nightly/cu124](https://download.pytorch.org/whl/nightly/cu124)
pip install -r requirements.txt
