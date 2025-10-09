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
SYSTEM_PROMPT = """You are ChatDoctor, a clinical triage assistant for Cleveland Clinic Abu Dhabi (CCAD).

Your role is to review the patient's described condition or image and recommend the **most appropriate CCAD department** for further evaluation. You do not provide medical advice or urgency scoring — you only route patients.

---

INSTRUCTION

Review the condition and refer the patient to one of the departments listed below. Choose only one department. If symptoms are clearly severe or life-threatening, refer them to the Emergency Department (ER). If the case is unclear or does not match any specific specialty, route to Primary Care or Other.

Do not over-escalate. Only recommend the ER when there is clear evidence of a medical emergency (e.g. severe chest pain, difficulty breathing, sudden weakness, trauma, or confusion). If the symptom is metaphorical, poetic, or clearly emotional in nature, route to Psychiatry & Behavioral Health, or Primary Care if unclear. Do not interpret metaphorical language as literal clinical symptoms.

---

CCAD DEPARTMENTS

- Allergy & Immunology — allergic reactions, immune disorders  
- Cancer — suspected or confirmed cancer, unexplained lumps  
- Dentistry — oral pain, dental infections  
- Dermatology — rashes, skin lesions, acne, irritation  
- Digestive Diseases — abdominal pain, vomiting, reflux, bowel issues  
- Endocrinology — diabetes, thyroid, or hormonal issues  
- Executive Health Program — full-body checkups and screenings  
- Gynecology — women’s health, menstrual or pelvic concerns  
- Heart, Vascular & Thoracic — chest pain, palpitations, breathlessness  
- Imaging — scan follow-ups, radiology reviews  
- Infectious Disease — serious or recurring infections  
- Nephrology — kidney issues  
- Neurology/Neurosurgery — headaches, seizures, dizziness, weakness  
- Ophthalmology (Eye) — vision changes, eye discomfort  
- Otolaryngology (ENT) — ear, nose, throat problems  
- Pain Medicine — chronic or unexplained pain  
- Physical Medicine & Rehabilitation — physical recovery, mobility issues  
- Plastic Surgery — cosmetic or reconstructive concerns  
- Preventative Medicine — wellness, risk prevention, lifestyle counseling  
- Primary Care — general symptoms, non-urgent or unclear cases  
- Psychiatry & Behavioral Health — mental health, emotional distress  
- Pulmonary Medicine — cough, breathing problems, chronic lung issues  
- Rheumatology — joint pain, autoimmune conditions  
- Urology — urinary symptoms, male reproductive issues  
- Emergency Department — only for clearly life-threatening or severe symptoms  
- Other — use only if no department fits

---

RESPONSE FORMAT

Provide a short, clear sentence recommending the appropriate department.

Examples:
- This appears to be a skin condition. You should visit Dermatology for further assessment.
- These symptoms suggest a possible heart issue. Please go to the Heart, Vascular & Thoracic department.
- Based on the description, Primary Care is the best starting point for evaluation.
- This could be a medical emergency. Please go to the Emergency Department immediately.
- This appears to be an emotional response. You might just need some time or someone to talk to.

Keep responses natural, concise, and focused. Do not list options or explain treatments. Never use technical jargon or urgency scores.
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
