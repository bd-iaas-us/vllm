#!/bin/bash

# Define variables
PORT=8000
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_MAX_LEN=30000
MODEL=$(echo "$MODEL" | xargs)
tp_size=2

request_rate_start=1
request_rate_end=8
request_rate_factor=2
prompt_len=1024

num_prompts=300


# PROMPT_LEN_START=1000
# PROMPT_LEN_END=9000
# PROMPT_LEN_STEP=1000

OUT_LEN_START=1
OUT_LEN_END=3001
OUT_LEN_STEP=1000

start_server_command="conda activate qian-vllm && vllm serve $MODEL --port $PORT --dtype float16 --enforce-eager --tensor-parallel-size $tp_size --enable-prefix-caching"
success_log="Application startup complete"
log_file="/tmp/server_log.txt"
timeout=120  # Timeout in seconds

# Function to check if the port is in use
check_port() {
    port=$1
    
    if [ -z "$port" ]; then
        echo "Error: No port number provided."
        echo "Usage: kill_processes_using_port <port>"
        return 1
    fi

    echo "Searching for processes using port $port..."

    # Find the PID of the process using the specified port
    pid=$(lsof -t -i:"$port")
    
    if [ -n "$pid" ]; then
        echo "Found process using port $port: PID $pid"
        
        # Kill all child processes and the parent process
        echo "Killing the process and its spawned processes..."
        pkill -TERM -P "$pid"
        kill -TERM "$pid" 2>/dev/null
        
        echo "All processes using port $port and their children have been terminated."
    else
        echo "No process found using port $port."
    fi
}

start_server() {
    local command=$1
    local log_file=$2
    local success_string=$3

    echo "Starting server with command: $command"
    bash -c "$command" > "$log_file" 2>&1 &

    # Wait for the server to start or timeout
    for ((i = 0; i < timeout; i++)); do
        if grep -q "$success_string" "$log_file"; then
            echo -e "\rServer started successfully!                "  # Clear the line
            echo $pid
            return 0
        fi
        sleep 1
        echo -ne "\rWaiting for vllm server to start... ($i seconds elapsed)"  # No newline
    done

    # If the loop completes, the server did not start in time
    echo -e "\rServer failed to start within $timeout seconds.          "  # Clear the line
    exit 1
}


# Function to kill the server
kill_gpu_processes() {
    echo "Killing all NVIDIA processes..."
    nvidia_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
    if [ -n "$nvidia_pids" ]; then
        echo "Found NVIDIA processes: $nvidia_pids"
        echo "$nvidia_pids" | xargs -n1 -r kill -9
        echo "All NVIDIA processes killed."
    else
        echo "No NVIDIA processes found."
    fi
}


# Clean up temporary files
# rm -rf /tmp/infinistore/* > /dev/null 2>&1 

echo "Running batch tests for model $MODEL, tp_size=$tp_size"

kill_gpu_processes

# Run benchmark for each prompt length
# for ((prompt_len=PROMPT_LEN_START; prompt_len<=PROMPT_LEN_END; prompt_len+=PROMPT_LEN_STEP)); do
for ((request_rate=request_rate_start; request_rate<=request_rate_end; request_rate*=request_rate_factor)); do
    for ((output_len=OUT_LEN_START; output_len<=OUT_LEN_END; output_len+=OUT_LEN_STEP)); do  
        if (( prompt_len + output_len > MODEL_MAX_LEN )); then
            continue
        fi

        echo "_____________________________________________________________________________"
        echo "Running benchmark with prompt_len=$prompt_len and output_len=$output_len, request_rate=$request_rate"
        echo 
        
        # Ensure the port is not in use
        check_port $PORT

        # Start the server
        start_server "$start_server_command" "$log_file" "$success_log"

                # Warm up the system
        curl http://10.192.18.145:$PORT/v1/completions -H "Content-Type: application/json" \
            -d '{"model": "'$MODEL'", "prompt": "San Francisco is a", "max_tokens": 100}'

        curl http://10.192.18.145:$PORT/v1/completions -H "Content-Type: application/json" \
            -d '{"model": "'$MODEL'", "prompt": "San Francisco is not a", "max_tokens": 100}'

        sleep 1

        echo 
        echo "Running benchmark..."


        python3 benchmarks/benchmark_serving.py \
            --port $PORT \
            --backend vllm \
            --dataset-name booksum \
            --dataset-path /data00/qian/git/vllm/booksum_data.csv \
            --model $MODEL \
            --endpoint /v1/completions \
            --gpu-metric-interval 0.5 \
            --request-rate $request_rate \
            --num-prompts $num_prompts \
            --booksum-output-len $output_len \
            --booksum-fix-prompt-len $prompt_len
        sleep 1

        kill_gpu_processes 
        pkill -f "vllm serve"
    done
done

echo "All tests completed."
