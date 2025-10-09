from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import ollama
import whisper
from piper import PiperVoice
import wave
import os
import uuid
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Create output directory and mount for static files
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Global conversation history
conversation_history = []

# System prompt for ChatDoctor
SYSTEM_PROMPT = """You are **ChatDoctor**, an AI clinician representing **Cleveland Clinic Abu Dhabi (CCAD)**.

🎯 Your mission:
Estimate an **Urgency Level (1–10)** and direct the patient to the correct **care pathway** or **CCAD institute**.

### 🔢 URGENCY SCALE
- 8–10 → **Emergency Department (ER / call 999)** – life-threatening or severe symptoms.
- 5–7  → **Outpatient / Specialty Institute** – needs in-person assessment soon.
- 3–4  → **Telehealth or Virtual Visit** – mild/moderate, safe to review virtually.
- 1–2  → **Self-care with watchouts** – reassure, monitor, and outline warning signs.

Keep your tone *calm, concise, and compassionate*.  
Use **3–5 short sentences**, natural human phrasing.  
Never list numbers, bullet points, or medications.  
Only say "999" for emergencies (no other numbers).

### 🏥 CCAD INSTITUTES (for routing)
If you detect relevant context, mention **the right CCAD institute** once by name:
- **Heart, Vascular & Thoracic Institute** → chest pain, palpitations, breathlessness.
- **Neurological Institute** → headache, dizziness, seizures, weakness, confusion.
- **Digestive Disease Institute** → abdominal pain, vomiting, reflux, bowel issues.
- **Cancer Institute** → suspected or known malignancy, unexplained lumps.
- **Medical Specialty Institute** → diabetes, hypertension, infections, chronic illness.
- **Integrated Surgical Institute** → post-op wound care, infections, trauma.
- **Diagnostics Institute** → lab results, imaging review, follow-up investigations.
- **Integrated Hospital Care Institute** → inpatient or chronic complex cases.

If unclear which institute fits, use **Telehealth** as default routing.
"""

# Global AI models
class AIModels:
    def __init__(self):
        print("Loading AI models...")
        self.whisper = whisper.load_model("base")
        self.mistral_model = "mistral"
        self.tts_voice_en = PiperVoice.load(
            "c:\\Users\\ihash\\Desktop\\clinapse\\clevland\\backend\\piper_models\\en_US-lessac-medium.onnx"
        )
        self.tts_voice_ar = PiperVoice.load(
            "c:\\Users\\ihash\\Desktop\\clinapse\\clevland\\backend\\piper_models\\ar_JO-kareem-medium.onnx"
        )
        print("✓ All models loaded (English + Arabic)")

models = AIModels()

async def stream_chatdoctor(condition: str):
    """Stream responses from Mistral model"""
    print("\n💬 ChatDoctor Response:\n")
    try:
        stream = ollama.chat(
            model="mistral",
            stream=True,
            options={"temperature": 0, "num_predict": 120},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Patient reports: {condition}. Please assess urgency and advise routing."}
            ]
        )

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                piece = chunk["message"]["content"]
                print(piece, end="", flush=True)
                yield piece

        print()

    except Exception as e:
        error_msg = f"⚠️ Ollama streaming failed: {e}"
        print(error_msg)
        yield error_msg

async def generate_events(condition: str):
    """Convert model output chunks to SSE format"""
    try:
        async for chunk in stream_chatdoctor(condition):
            if chunk:
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/chatdoctor")
async def chatdoctor(request: Request):
    """Text-based chat endpoint"""
    data = await request.json()
    condition = data.get("condition", "")
    print(f"\n📨 Received condition: {condition}")
    
    return StreamingResponse(
        generate_events(condition),
        media_type="text/event-stream"
    )

@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    """Voice-based chat endpoint"""
    try:
        # Save uploaded audio
        audio_id = str(uuid.uuid4())
        input_path = f"outputs/input_{audio_id}.wav"

        with open(input_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        # Whisper transcription
        print(f"\n📝 Transcribing audio from: {input_path}")
        try:
            result = models.whisper.transcribe(input_path)
            print(f"Raw Whisper result: {result}")  # Debug print
            
            transcription = result["text"].strip()
            detected_language = result.get("language", "en")

            print(f"Detected language: {detected_language}")
            print(f"User said: {transcription}")

            # More detailed validation
            if not transcription:
                raise ValueError("Empty transcription returned from Whisper")
            
            if len(transcription) < 3:
                raise ValueError(f"Transcription too short: '{transcription}'")

        except Exception as e:
            print(f"❌ Transcription error: {str(e)}")
            if os.path.exists(input_path):
                os.remove(input_path)
            return JSONResponse({
                "success": False,
                "error": f"Speech recognition failed: {str(e)}"
            }, status_code=400)

        # Add to conversation history with language instruction
        if detected_language == "ar":
            conversation_history.append({
                'role': 'user',
                'content': f"{transcription}\n\n(Respond in Arabic in 2-3 sentences max for voice conversation)"
            })
        else:
            conversation_history.append({
                'role': 'user',
                'content': f"{transcription}\n\n(Respond in English in 2-3 sentences max for voice conversation)"
            })

        # Process with Mistral
        print(f"\n🤖 Getting Mistral response...")
        print(f"📚 Conversation history length: {len(conversation_history)} messages")

        response = ollama.chat(
            model=models.mistral_model,
            messages=conversation_history
        )

        assistant_response = response['message']['content']
        print(f"Assistant: {assistant_response}")

        # Add to conversation history
        conversation_history.append({
            'role': 'assistant',
            'content': assistant_response
        })

        # Generate speech response
        print(f"\n🔊 Generating speech...")
        output_audio_path = f"outputs/response_{audio_id}.wav"
        
        # Select voice based on detected language
        tts_voice = models.tts_voice_ar if detected_language == "ar" else models.tts_voice_en

        audio_chunks = list(tts_voice.synthesize(assistant_response))
        with wave.open(output_audio_path, "wb") as wav_file:
            if audio_chunks:
                first_chunk = audio_chunks[0]
                wav_file.setnchannels(first_chunk.sample_channels)
                wav_file.setsampwidth(first_chunk.sample_width)
                wav_file.setframerate(first_chunk.sample_rate)
                for chunk in audio_chunks:
                    wav_file.writeframes(chunk.audio_int16_bytes)

        # Cleanup and return
        os.remove(input_path)
        print(f"✅ Complete!")

        return JSONResponse({
            "success": True,
            "transcription": transcription,
            "text": assistant_response,
            "audio_url": f"/outputs/response_{audio_id}.wav",
            "language": detected_language
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/clear-history")
async def clear_history():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return {"success": True, "message": "History cleared"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "models": "loaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
