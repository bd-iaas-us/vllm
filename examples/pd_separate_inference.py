from vllm import LLM, SamplingParams
import time
import os
import subprocess
import json

# model_name = "huggyllama/llama-7b"
model_name = "meta-llama/Meta-Llama-3-8B"

prompts_1 = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]

prompts_2 = [
    # "The capital of France is",
    # "The future of AI is",
    "Hello, my name is",
    # "The president of the United States is",
]

# os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

prefill_params_dict = {
    "model": model_name,
    "prompts": prompts_1,
}

decode_params_dict = {
    "model": model_name,
    "prompts": prompts_2,
}

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

prefill_script_path = os.path.join(script_dir,
                                   "prefill_pd_separate_inference.py")
decode_script_path = os.path.join(script_dir,
                                  "decode_pd_separate_inference.py")

try:
    print("prefill stage start .....")
    subprocess.run(
        ["python", prefill_script_path,
         json.dumps(prefill_params_dict)],
        check=True)
    print("prefill stage ends")

    print("decode stage start .....")
    subprocess.run(
        ["python", decode_script_path,
         json.dumps(decode_params_dict)],
        check=True)
    print("decode stage ends")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
