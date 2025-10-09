import warnings
warnings.filterwarnings("ignore")

import whisper
import ollama
from piper import PiperVoice
import wave
import os
import sys

class SimpleHospitalAI:
    def __init__(self):
        """Simple: Whisper → Mistral → Piper"""
        print("="*50)
        print("Loading Hospital AI System...")
        print("="*50)
        
        # Load Whisper
        print("Loading Whisper...")
        self.whisper = whisper.load_model("base")
        print("✓ Whisper ready")
        
        # Mistral model
        self.mistral_model = "my-mistral"
        print("✓ Mistral ready")
        
        # Load Piper TTS
        print("Loading Piper voice...")
        tts_path = "/mnt/c/Users/yasin/Desktop/YasinAi/venv/piper_models/en_US-lessac-medium.onnx"
        self.tts_voice = PiperVoice.load(tts_path)
        print("✓ Piper ready")
        
        print("="*50)
        print("System Ready!\n")
    
    def run(self, audio_file):
        """Run the simple pipeline"""
        
        # Check if file exists
        if not os.path.exists(audio_file):
            print(f"❌ File not found: {audio_file}")
            return
        
        # Step 1: Whisper - Transcribe audio
        print("\n📝 STEP 2: Transcribing with Whisper...")
        result = self.whisper.transcribe(audio_file)
        transcription = result["text"]
        print(f"Transcription: {transcription}")
        
        # Step 2: Mistral - Format as clinical note
        print("\n🤖 STEP 3: Formatting with Mistral...")
        response = ollama.chat(
            model=self.mistral_model,
            messages=[
                {
                    'role': 'user', 
                    'content': transcription
                }
            ]
        )
        formatted_note = response['message']['content']
        print(f"Formatted Note:\n{formatted_note}")
        
        # Step 3: Piper - Convert to speech
        print("\n🔊 STEP 4: Generating speech with Piper...")
        output_file = "output_clinical_note.wav"
        
        # Generate audio
        audio_chunks = list(self.tts_voice.synthesize(formatted_note))
        
        # Save to WAV
        with wave.open(output_file, "wb") as wav_file:
            if audio_chunks:
                first_chunk = audio_chunks[0]
                wav_file.setnchannels(first_chunk.sample_channels)
                wav_file.setsampwidth(first_chunk.sample_width)
                wav_file.setframerate(first_chunk.sample_rate)
                
                for chunk in audio_chunks:
                    wav_file.writeframes(chunk.audio_int16_bytes)
        
        print(f"✅ Done! Audio saved to: {output_file}")
        print("\n" + "="*50)
        print("Pipeline Complete!")
        print("="*50)

if __name__ == "__main__":
    # Check if audio file argument is provided
    if len(sys.argv) < 2:
        print("Usage: python simple_hospital_ai.py <audio_file>")
        print("Example: python simple_hospital_ai.py recording.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    ai = SimpleHospitalAI()
    ai.run(audio_file)