from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import asyncio
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# URLs for the two vLLM processes
VLLM_1_URL = "http://10.192.18.145:8000/v1/completions"
VLLM_2_URL = "http://10.192.24.218:8000/v1/completions"

async def stream_vllm_response(vllm_url, req_data):
    """Asynchronously streams the response from a vLLM process."""
    logger.info(f"Sending request to {vllm_url} with data: {req_data}")
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", vllm_url, json=req_data) as response:
            response.raise_for_status()  # Raise error if response is not successful
            async for chunk in response.aiter_bytes():
                yield chunk
            logger.info(f"Received complete response from {vllm_url}")

def remove_first_word(text):
    """Removes the first word from the text."""
    logger.info("Modifying response from vLLM-2 by removing the first word.")
    return ' '.join(text.split()[1:])

@app.post("/generate")
async def proxy_request(request: Request):
    # Extract the incoming request JSON data
    req_data = await request.json()  
    logger.info(f"Received new request with data: {req_data}")
    
    # Ensure that "model", "prompt", and "max_tokens" are in the request data
    if not all(key in req_data for key in ["model", "prompt", "max_tokens"]):
        error_message = "Missing 'model', 'prompt', or 'max_tokens' in request body"
        logger.error(error_message)
        return {"error": error_message}

    # Forward the request to vLLM-1 first, then to vLLM-2
    async def generate_stream():
        try:
            # Step 1: Stream response from vLLM-1
            logger.info(f"Forwarding request to vLLM-1 ({VLLM_1_URL})")
            async for chunk in stream_vllm_response(VLLM_1_URL, req_data):
                yield chunk  # Stream response from vLLM-1 directly to the user

            # Step 2: After vLLM-1 response finishes, get response from vLLM-2
            logger.info(f"Forwarding request to vLLM-2 ({VLLM_2_URL})")
            full_vllm2_response = b""
            async for chunk in stream_vllm_response(VLLM_2_URL, req_data):
                full_vllm2_response += chunk

            # Step 3: Modify the response from vLLM-2 (remove first word)
            full_vllm2_text = full_vllm2_response.decode('utf-8')
            modified_vllm2_text = remove_first_word(full_vllm2_text)
            
            # Step 4: Stream the modified response to the user
            logger.info("Streaming modified response from vLLM-2 to the user.")
            yield modified_vllm2_text.encode('utf-8')
        
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            yield f"Error: {str(e)}".encode('utf-8')

    # Use StreamingResponse to stream the response back to the client
    return StreamingResponse(generate_stream(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
