from vllm import LLM, SamplingParams
import os
import json
import sys

params_json = sys.argv[1]
params_dict = json.loads(params_json)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
prompts = params_dict["prompts"]
model_name = params_dict["model"]

print("Prefill Stage: initilizeing llm engine.....")

llm = LLM(model=model_name, enforce_eager=True)

print(" Prefill Stage: inference start ...")
os.environ["pd_separate_stage"] = "decode"

outputs = llm.generate(prompts, sampling_params)

print("decode stage ends")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")