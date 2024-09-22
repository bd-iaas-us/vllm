import torch
import socket
import os
import subprocess
import time
import signal

model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Meta-Llama-3-8B"
prefill_log = "/tmp/vllm1.log"
decode_log = "/tmp/vllm2.log"

def check_ports(ports=[8000, 8001]):
    for port in ports:
        if not isinstance(port, int):
            raise Exception("port must be an integer")
        if port < 1024 or port > 65535:
            raise Exception("port must be between 1024 and 65535")
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('localhost', port)) == 0:
                raise Exception(f"port {port} is already in use")

def start_vllm_server(model_name, gpu_count, ports):
    # Ensure there are at least two GPUs
    if gpu_count < 2:
        raise Exception("Requires at least 2 GPUs to run the processes separately.")
    
    # Calculate the first half and second half of GPUs
    half_gpu_count = gpu_count // 2
    prefill_gpus = ",".join(str(i) for i in range(half_gpu_count))  # First half GPUs
    decode_gpus = ",".join(str(i) for i in range(half_gpu_count, gpu_count))  # Second half GPUs
    
    # Clear out the logs before starting
    open(prefill_log, 'w').close()
    open(decode_log, 'w').close()

    # Start the prefill process (first half of GPUs)
    prefill_env = os.environ.copy()
    prefill_env['CUDA_VISIBLE_DEVICES'] = prefill_gpus
    prefill_env['pd_separate_stage'] = 'prefill'
    
    prefill_command = f"/opt/conda/envs/qian/bin/vllm serve --enforce-eager {model_name} --port {ports[0]}"
    with open(prefill_log, 'w') as log_file:
        subprocess.Popen(prefill_command, shell=True, env=prefill_env,
                         stdout=log_file, stderr=log_file, close_fds=True)
    print(f"Starting prefill process on port {ports[0]} with GPUs {prefill_gpus}. Logs are being written to {prefill_log}")

    # Start the decode process (second half of GPUs)
    decode_env = os.environ.copy()
    decode_env['CUDA_VISIBLE_DEVICES'] = decode_gpus
    decode_env['pd_separate_stage'] = 'decode'
    
    decode_command = f"/opt/conda/envs/qian/bin/vllm serve --enforce-eager {model_name} --port {ports[1]}"
    with open(decode_log, 'w') as log_file:
        subprocess.Popen(decode_command, shell=True, env=decode_env,
                         stdout=log_file, stderr=log_file, close_fds=True)
    print(f"Starting decode process on port {ports[1]} with GPUs {decode_gpus}. Logs are being written to {decode_log}")

def wait_for_vllm_ready(timeout=60):
    target_string = "Application startup complete."
    start_time = time.time()
    prefill_ready = False
    decode_ready = False
    prefill_f = None
    decode_f = None

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise RuntimeError(f"vllm ervers did not start within {timeout} seconds.")

        # Try to open prefill log file if not already opened
        if prefill_f is None:
            if os.path.exists(prefill_log):
                prefill_f = open(prefill_log, 'r')
                prefill_f.seek(0, 2)  # Move to the end of the file
            else:
                print(f"Prefill log file {prefill_log} not found.")

        # Try to open decode log file if not already opened
        if decode_f is None:
            if os.path.exists(decode_log):
                decode_f = open(decode_log, 'r')
                decode_f.seek(0, 2)  # Move to the end of the file
            else:
                print(f"Decode log file {decode_log} not found.")

        # Read new lines from prefill log
        if prefill_f is not None:
            prefill_lines = prefill_f.readlines()
            if prefill_lines:
                if any(target_string in line for line in prefill_lines):
                    prefill_ready = True
                    print("~~~~~Prefill server is ready.~~~~~")
            #     else:
            #         print("Prefill server not ready yet, last log line:", prefill_lines[-1].strip())
            # else:
            #     print("No new lines in prefill server log.")

        # Read new lines from decode log
        if decode_f is not None:
            decode_lines = decode_f.readlines()
            if decode_lines:
                if any(target_string in line for line in decode_lines):
                    decode_ready = True
                    print("-----Decode server is ready.-----")
            #     else:
            #         print("Decode server last log line:", decode_lines[-1].strip())
            # else:
            #     print("No new lines in decode server log.")

        # Check if both servers are ready
        if prefill_ready and decode_ready:
            print("Servers are ready to take inputs.")
            if prefill_f:
                prefill_f.close()
            if decode_f:
                decode_f.close()
            return

        time.sleep(5)  # Wait before checking again
        print("--Waiting for servers to be ready...")

def start_infinity_store():
    binary_path = "/home/ubuntu/qian/git//infinity/src/infinity_server"
    # Check if the binary is running
    try:
        # Find all PIDs matching the binary path
        pids = subprocess.check_output(["pgrep", "-f", binary_path]).decode().split()
        # Terminate each running instance
        for pid in pids:
            os.kill(int(pid), signal.SIGTERM)
    except subprocess.CalledProcessError:
        # No running instances found
        pass
    # Start the binary in the background
    subprocess.Popen([binary_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def kill_vllm_processes():
    binary_path = "/opt/conda/envs/qian/bin/vllm"
    try:
        # Find all PIDs matching the binary path
        pids = subprocess.check_output(["pgrep", "-f", binary_path]).decode().split()
        if not pids:
            print(f"No running processes found for {binary_path}")
            return
        # Terminate each running instance
        for pid in pids:
            os.kill(int(pid), signal.SIGTERM)
            print(f"Killed process {pid} running {binary_path}")
    except subprocess.CalledProcessError:
        # No running instances found
        print(f"No running processes found for {binary_path}")

if __name__ == "__main__":

    ports = [8000, 8001]

    # Get the GPU count from torch
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count < 2:
        raise Exception("Requires at least 2 GPUs to run")

    # Check that the ports are available (this function can be reused)
    check_ports(ports)

    # kill_vllm_processes()

    # Start the VLLM server processes on different sets of GPUs
    start_vllm_server(model_name, gpu_count, ports)

    wait_for_vllm_ready()

    start_infinity_store()
