#!/bin/bash

# Define variables
PORT=8000
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
MODEL=$(echo "$MODEL" | xargs)


PROMPT_LEN_START=1000
PROMPT_LEN_END=32000
PROMPT_LEN_STEP=1000

rm -rf /tmp/infinistore/* > /dev/null 2>&1 

# Warm up the system
curl http://10.192.18.145:$PORT/v1/completions -H "Content-Type: application/json" \
    -d '{"model": "'$MODEL'", "prompt": "San Francisco is a", "max_tokens": 100}'

curl http://10.192.18.145:$PORT/v1/completions -H "Content-Type: application/json" \
    -d '{"model": "'$MODEL'", "prompt": "San Francisco is not a", "max_tokens": 100}'

sleep 1

# Run benchmark for each prompt length
for ((prompt_len=$PROMPT_LEN_START; prompt_len<=$PROMPT_LEN_END; prompt_len+=$PROMPT_LEN_STEP)); do
    echo 
    echo
    echo "prompt = $prompt_len"
    python3 benchmarks/benchmark_serving.py \
        --port $PORT \
        --backend vllm \
        --dataset-name booksum \
        --dataset-path /data00/qian/git/vllm/booksum_data.csv \
        --model $MODEL \
        --endpoint /v1/completions \
        --gpu-metric-interval 0.5 \
        --request-rate 0.5 \
        --num-prompts 1 \
        --booksum-output-len 1 \
        --booksum-fix-prompt-len $prompt_len
    sleep 1
    
done