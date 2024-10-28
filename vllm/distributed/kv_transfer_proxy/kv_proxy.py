from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import asyncio

app = FastAPI()

# URLs for the two vLLM processes
VLLM_1_URL = "http://10.192.18.145:8000/v1/completions"
VLLM_2_URL = "http://10.192.24.218:8000/v1/completions"

async def stream_vllm_response(vllm_url, req_data):
    """Asynchronously streams the response from a vLLM process."""
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", vllm_url, json=req_data) as response:
            response.raise_for_status()  # Ensure successful response
            async for chunk in response.aiter_bytes():
                yield chunk

@app.post("/v1/completions")
async def proxy_request(request: Request):
    # Extract the incoming request JSON data
    req_data = await request.json()  
    
    # Ensure that "model", "prompt", and "max_tokens" are in the request data
    if not all(key in req_data for key in ["model", "prompt", "max_tokens"]):
        return {"error": "Missing 'model', 'prompt', or 'max_tokens' in request body"}

    # Forward the request to vLLM-1 (but ignore its response)
    async def generate_stream():
        # Step 1: Send request to vLLM-1 (we do not yield its response)
        async for _ in stream_vllm_response(VLLM_1_URL, req_data):
            pass  # Skip vLLM-1 response chunks entirely

        # Step 2: Send the request to vLLM-2 and stream its response as-is
        async for chunk in stream_vllm_response(VLLM_2_URL, req_data):
            yield chunk  # Stream response from vLLM-2 directly to the user

    # Use StreamingResponse to stream the response back to the client
    return StreamingResponse(generate_stream(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
