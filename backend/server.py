from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mistral import stream_chatdoctor  # Updated import
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

async def generate_events(condition: str):
    """Convert model output chunks to SSE format."""
    try:
        async for chunk in stream_chatdoctor(condition):
            if chunk:  # Only yield non-empty chunks
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/chatdoctor")
async def chatdoctor(request: Request):
    data = await request.json()
    condition = data.get("condition", "")
    print(f"\n📨 Received condition: {condition}")
    
    return StreamingResponse(
        generate_events(condition),
        media_type="text/event-stream"
    )
