# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# model_id = "meta-llama/Llama-Guard-3-8B-INT8"
# device = "cuda"
# dtype = torch.bfloat16

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device, quantization_config=quantization_config)

# def moderate(chat):
#     input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
#     output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
#     prompt_len = input_ids.shape[-1]
#     return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

# output = moderate([
#     "I am a student study Japanese."
# ])

# print(output)
from transformers import AutoTokenizer, AutoModelForCausalLM

tok_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = "LsTam/Mistral-7B-Instruct-v0.1-8bit"

tokenizer = AutoTokenizer.from_pretrained(tok_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #use_flash_attention_2=True,
    )

prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

prompt = 'Please show how binary search works in Python.'


inputt = prompt_template.format(instruction= prompt)

inputt = "my name is"
input_ids = tokenizer(inputt, return_tensors="pt").input_ids.to("cuda")

output1 = model.generate(input_ids, max_length=512)
input_length = input_ids.shape[1]
output1 = output1[:, input_length:]
output = tokenizer.decode(output1[0])

print("\n ~~~~~~~~~~~~~~~~~~")
print(output)

