from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    #"[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
    #"<<SYS>>\nAnswer the following Grade School Math problem.\n<</SYS>>\n[INST] How much is five plus six? [/INST]\n",
     f'''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHow much is five plus three?<|im_end|>\n<|im_start|>assistant\n'''
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128) #,
                        #stop_token_ids=[32003])

# Create an LLM.
llm = LLM(model="meta-llama/Llama-2-7b-hf")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    # print(output)
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}, Generated text: {generated_text}")


