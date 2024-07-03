from vllm import LLM, SamplingParams
import time

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
start = time.time()
#llm = LLM(model="meta-llama/Llama-2-7b-hf", cpu_offload_weight=False)
#llm = LLM(model="mistralai/Mistral-7B-v0.1", max_model_len=2048, cpu_offload_weight=False)
#llm = LLM(model="meta-llama/Llama-2-13b-hf", cpu_offload_weight=True)
llm = LLM(model="meta-llama/Llama-2-13b-hf", cpu_offload_weight=False, tensor_parallel_size=4)
#llm = LLM(model="meta-llama/Llama-2-70b-hf", cpu_offload_weight=True)
#llm = LLM(model="meta-llama/Llama-2-70b-hf", cpu_offload_weight=False, tensor_parallel_size=4)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

time_cost = time.time() - start
print(f"time_track ===== : end-to-end time cost  {time_cost}")
