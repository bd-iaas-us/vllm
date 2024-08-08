from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer

# Sample prompts.
prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

prompt = 'Please show how binary search works in Python.'
inputt = prompt_template.format(instruction= prompt)
prompts =[
    "The capital of France is",
]

tok_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(tok_name)
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.

#llm = LLM(model="lllyasviel/omost-llama-3-8b-4bits", dtype=torch.bfloat16, trust_remote_code=True,\
#llm = LLM(model="meta-llama/Llama-Guard-3-8B-INT8", trust_remote_code=True,\
llm = LLM(model="LsTam/Mistral-7B-Instruct-v0.1-8bit", trust_remote_code=True, tokenizer="mistralai/Mistral-7B-Instruct-v0.1",
           quantization="bitsandbytes", load_format="bitsandbytes", gpu_memory_utilization=0.8, enforce_eager=True, max_model_len=4096)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
