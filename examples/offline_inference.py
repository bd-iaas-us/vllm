from vllm import LLM, SamplingParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ['HF_HOME'] = '/data00/tony'
os.environ['TRANSFORMERS_CACHE'] = '/data00/tony'
os.environ['HF_DATASETS_CACHE'] = '/data00/tony'
os.environ['CUDA_LAUNCH_BLOCKING']= '1'
# export HF_HOME=/data00/tony
# export TRANSFORMERS_CACHE=/data00/tony
# export HF_DATASETS_CACHE=/data00/tony
# export HF_TOKEN=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
# export TMPDIR=/data00/tony/tmp
# export CUDACXX=/usr/local/cuda-12.1/bin/nvcc

# Sample prompts.
prompts = [
    #"Hello, my name is Tony, and I'm thrilled to have the opportunity to introduce myself to you. I am a motivated and enthusiastic individual with a passion for technology, marketing, finance. Professionally, I have worked in the technology industry for five years. These experiences have equipped me with valuable skills in leadership, problem-solving, communication.Outside of work, I enjoy hiking, photography and reading, which help me maintain a balanced and fulfilling lifestyle. I also believe in giving back to the community and have volunteered with hunger.",
    #"Hello, my name is Tony, I like everything in",
    # "Hello, my name is",
    # "The president of the United States is",
    "Hello, my name is Tony, and I'm thrilled to have the opportunity to introduce myself to you. I am a motivated and enthusiastic individual with a passion for technology, marketing, finance. ",
    "The president of the United States is Donald Trump, who has been involved in a lawsuit, and he has been regarded ",
    "The capital of France is",
    "The future of AI is",
    # "Hello, my name is Tony, I like everything in",
    # "Hello, my name is",
    # "The president of the United States is Trump",
    # "The capital of France is where ",
    # "The future of AI is now",
    # "I love Japanese",
    # "I love Chinese",
    # "I love American",
    # "I love African",
    # "I love White",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
#llm = LLM(model="facebook/opt-125m", tensor_parallel_size = 1, kv_cache_dtype='auto')
#llm = LLM(model="facebook/opt-125m", tensor_parallel_size = 1, max_model_len=100, kv_cache_dtype='auto')
llm = LLM(model="facebook/opt-125m", tensor_parallel_size = 1, max_model_len=100, kv_cache_dtype='auto', sparse_kv_cache_type='h2o')
#llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", tensor_parallel_size = 1, gpu_memory_utilization=0.95, max_model_len=100, kv_cache_dtype='auto', sparse_kv_cache_type='h2o')
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# export PATH="/data00/miniconda3/envs/tony-env/bin:$PATH"
# export PATH="/usr/local/cuda/bin:/data00/miniconda3/envs/tony-env/bin:/data00/miniconda3/envs/qian/bin:/data00/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
# /data00/miniconda3/envs/tony-bd-vllm-us/bin:/usr/local/bin:/root/.bun/bin:/root/.cargo/bin:/data00/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda-12.1/bin
# /usr/local/cuda/bin:/data00/miniconda3/envs/tony-env/bin:/data00/miniconda3/envs/qian/bin:/data00/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# export CC=gcc-9
# export CXX=g++-9
# conda install gcc_linux-64=9
# access_token = "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"