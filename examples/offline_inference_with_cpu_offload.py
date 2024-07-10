from vllm import LLM, SamplingParams

import vllm.model_executor.layers.linear as linear_layers
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

start = time.time()
# Create an LLM.
print("=== time_track ===: loading model...")
linear_layers.total_weight_load_time =  0
linear_layers.total_matrix_compute_time = 0
# linear_layers.weight_move_cost_0 = 0
# linear_layers.weight_move_cost_1 = 0
# linear_layers.weight_move_cost_2 = 0
# linear_layers.weight_move_cost_3 = 0

#llm = LLM(model="meta-llama/Llama-2-7b-hf", tensor_parallel_size=1)
#llm = LLM(model="meta-llama/Llama-2-7b-hf", gpu_weight_memory_percentage=0.6, tensor_parallel_size=1)
#llm = LLM(model="meta-llama/Llama-2-13b-hf", tensor_parallel_size=4)

#llm = LLM(model="meta-llama/Llama-2-13b-hf", gpu_weight_memory_percentage=0.25, tensor_parallel_size=4)
llm = LLM(model="meta-llama/Llama-2-70b-hf", gpu_weight_memory_percentage=0.001, tensor_parallel_size=1)

print("=== time_track ===: Done running the model for memory profiling.")
print("=== time_track ===: Total weight load time: ", linear_layers.total_weight_load_time)
print("=== time_track ===: Total matrix compute time: ", linear_layers.total_matrix_compute_time)
print("=== time_track ===: Total time to load the model: ", time.time()-start)

#------------------
start = time.time()
print("=== time_track ===: Running the model ..")
linear_layers.total_weight_load_time =  0
linear_layers.total_matrix_compute_time = 0
# linear_layers.weight_move_cost_0 = 0
# linear_layers.weight_move_cost_1 = 0
# linear_layers.weight_move_cost_2 = 0
# linear_layers.weight_move_cost_3 = 0

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
print("=========================")
print("=== time_track ===: Done running the model for memory profiling.")
print("=== time_track ===: Total weight load time: ", linear_layers.total_weight_load_time)
print("=== time_track ===: Total matrix compute time: ", linear_layers.total_matrix_compute_time)

# print("=== time_track ===: tp 0 weight move: ", linear_layers.weight_move_cost_0)
# print("=== time_track ===: tp 1 weight move: ", linear_layers.weight_move_cost_1)
# print("=== time_track ===: tp 2 weight move: ", linear_layers.weight_move_cost_2)
# print("=== time_track ===: tp 3 weight move: ", linear_layers.weight_move_cost_3)


print("=== time_track ===: total time to execute the model: ", time.time()-start)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
