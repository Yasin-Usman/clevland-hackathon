from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import whisper
import ollama
from piper import PiperVoice
import wave
import os
import uuid
from datetime import datetime

app = FastAPI()

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Global AI models (load once at startup)


class AIModels:
    def __init__(self):
        print("Loading AI models...")
        self.whisper = whisper.load_model("base")
        self.mistral_model = "my-mistral"

        # Load English TTS
        self.tts_voice_en = PiperVoice.load(
            "c:\\Users\\ihash\\Desktop\\clinapse\\clevland\\backend\\piper_models\\en_US-lessac-medium.onnx"
        )

        # Load Arabic TTS
        self.tts_voice_ar = PiperVoice.load(
            "c:\\Users\\ihash\\Desktop\\clinapse\\clevland\\backend\\piper_models\\ar_JO-kareem-medium.onnx"
        )

        print("✓ All models loaded (English + Arabic)")


models = AIModels()

# Store conversation history
conversation_history = []


@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    """
    Main endpoint: Receives audio → Whisper → Mistral → Piper
    """
    try:
        # Save uploaded audio
        audio_id = str(uuid.uuid4())
        input_path = f"outputs/input_{audio_id}.wav"

        with open(input_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        print(f"\n📝 Transcribing audio...")
        # Step 1: Whisper transcription with language detection
        result = models.whisper.transcribe(input_path)
        transcription = result["text"].strip()
        detected_language = result.get(
            "language", "en")  # Get detected language

        print(f"Detected language: {detected_language}")
        print(f"User said: {transcription}")

        # Ignore empty or too short transcriptions
        if not transcription or len(transcription) < 3:
            os.remove(input_path)
            return JSONResponse({
                "success": False,
                "error": "No speech detected"
            }, status_code=400)

        # Add to conversation history with language instruction
        if detected_language == "ar":
            # Arabic detected - instruct to respond in Arabic
            conversation_history.append({
                'role': 'user',
                'content': f"{transcription}\n\n(Respond in Arabic in 2-3 sentences max for voice conversation)"
            })
        else:
            # English or other - instruct to respond in English
            conversation_history.append({
                'role': 'user',
                'content': f"{transcription}\n\n(Respond in English in 2-3 sentences max for voice conversation)"
            })

        print(f"\n🤖 Getting Mistral response...")
        print(
            f"📚 Conversation history length: {len(conversation_history)} messages")

        # Step 2: Mistral response (with full conversation context)
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

        print(f"\n🔊 Generating speech...")
        # Step 3: Piper TTS - Select voice based on language
        output_audio_path = f"outputs/response_{audio_id}.wav"

        # Choose TTS voice based on detected language
        if detected_language == "ar":
            print("Using Arabic voice...")
            tts_voice = models.tts_voice_ar
        else:
            print("Using English voice...")
            tts_voice = models.tts_voice_en

        # Synthesize audio
        audio_chunks = list(tts_voice.synthesize(assistant_response))

        with wave.open(output_audio_path, "wb") as wav_file:
            if audio_chunks:
                first_chunk = audio_chunks[0]
                wav_file.setnchannels(first_chunk.sample_channels)
                wav_file.setsampwidth(first_chunk.sample_width)
                wav_file.setframerate(first_chunk.sample_rate)

                for chunk in audio_chunks:
                    wav_file.writeframes(chunk.audio_int16_bytes)

        print(f"✅ Complete!")

        # Clean up input file
        os.remove(input_path)

        return JSONResponse({
            "success": True,
            "transcription": transcription,
            "text": assistant_response,
            "audio_url": f"/outputs/response_{audio_id}.wav",
            "language": detected_language  # Send language to frontend
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
