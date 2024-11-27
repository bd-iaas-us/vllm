import os
import subprocess


# Parameters to vary
output_lens = [1, 100,1000]
prompt_lens = [500, 1000, 2000, 5000, 10000]
req_rates = [1,2,4,6,8,10]
num_prompts = 100


# change the following per the model running in vllms
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_max_len = 32786
escaped_model_name = model_name.replace("/", "_")


short_prompt_dataset = "/data00/qian/git/vllm/booksum_data.csv"
long_prompt_dataset = "/data00/qian/git/vllm/booksum_prompt_15k_to_50k.csv"

# Fixed parameters
base_command = [
    "CUDA_VISIBLE_DEVICES=0,1",
    "python3",
    "benchmarks/benchmark_serving.py",
    "--port", "8000",
    "--backend", "vllm",
    "--dataset-name", "booksum",
    "--model", model_name,
    "--num-prompts", str(num_prompts),
    "--endpoint", "/v1/completions",
    "--gpu-metric-interval", "0.5"
]

# Directory for logs
log_dir = os.path.join("/data00/qian/benchmark_logs", escaped_model_name)
os.makedirs(log_dir, exist_ok=True)

# Run combinations of parameters
for output_len in output_lens:
    for prompt_len in prompt_lens:
        for rate in req_rates:
            # 
            if output_len + prompt_len > 32000:
                continue
        # Build the command
            command = base_command + [
                "--booksum-output-len", str(output_len),
                "--booksum-fix-prompt-len", str(prompt_len),
                "--dataset-path", long_prompt_dataset if prompt_len > 15000 else short_prompt_dataset,
                "--request-rate", str(rate),
            ]
            
            # Generate the log file name
            log_file = os.path.join(
                log_dir,
                f"out_{output_len}_prompt_{prompt_len}_rate_{rate}.log"
            )
            if os.path.exists(log_file):
                print(f"Skipping command with output-len={output_len}, prompt-len={prompt_len}, rate={rate}. Log file already exists.")
                continue
            print("\n ---------------------------------------------------------------------- \n")
            print(f"Running command with output-len={output_len}, prompt-len={prompt_len}, rate={rate}.")
            with open(log_file, "w") as log:
                # Open subprocess and stream output to both log and console
                process = subprocess.Popen(" ".join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in process.stdout:
                    decoded_line = line.decode('utf-8')
                    print(decoded_line, end='')  # Print to console
                    log.write(decoded_line)     # Write to log file
                process.wait()
                
                if process.returncode != 0:
                    print(f"Command failed for output-len={output_len}, prompt-len={prompt_len}. See {log_file} for details.")

print(f"All commands executed. Logs are stored in the '{log_dir}' directory.")
