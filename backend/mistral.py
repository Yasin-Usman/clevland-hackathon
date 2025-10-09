import time
import sys
import os
from functools import lru_cache
from transformers import (
	AutoTokenizer, 
	AutoModelForCausalLM, 
	pipeline,
	BitsAndBytesConfig
)
import torch
import logging

# Set up logging to reduce noise
logging.getLogger("transformers").setLevel(logging.WARNING)

SYSTEM_PROMPT = """You are ChatDoctor, a clinical triage assistant for Cleveland Clinic Abu Dhabi (CCAD).

Your role is to review the patient's described condition or image and recommend the **most appropriate CCAD department** for further evaluation. You do not provide medical advice or urgency scoring — you only route patients.

---

INSTRUCTION

Review the condition and refer the patient to one of the departments listed below. Choose only one department. If symptoms are clearly severe or life-threatening, refer them to the Emergency Department (ER). If the case is unclear or does not match any specific specialty, route to Primary Care or Other.

Do not over-escalate. Only recommend the ER when there is clear evidence of a medical emergency (e.g. severe chest pain, difficulty breathing, sudden weakness, trauma, or confusion).

---

CCAD DEPARTMENTS

- Allergy & Immunology — allergic reactions, immune disorders  
- Cancer — suspected or confirmed cancer, unexplained lumps  
- Dentistry — oral pain, dental infections  
- Dermatology — rashes, skin lesions, acne, irritation  
- Digestive Diseases — abdominal pain, vomiting, reflux, bowel issues  
- Endocrinology — diabetes, thyroid, or hormonal issues  
- Executive Health Program — full-body checkups and screenings  
- Gynecology — women's health, menstrual or pelvic concerns  
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

Keep responses natural, concise, and focused. Do not list options or explain treatments. Never use technical jargon or urgency scores.
"""

# MODEL_NAME = "microsoft/DialoGPT-small"
MODEL_NAME = "Sreehariiii/BioGPT-4bit-quantized-version"  # BioGPT 4-bit quantized
MODEL_CACHE_DIR = "./model_cache"
_model_pipeline = None
_tokenizer = None

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


def preload_model():
	"""Preload the model."""
	global _model_pipeline, _tokenizer
	
	if _model_pipeline is None:
		try:
			print("🔄 Loading BioGPT 4-bit quantized model...")
			print(f"📁 Model will be cached in: {MODEL_CACHE_DIR}")
			print("⏳ This may take several minutes on first download...")

			os.makedirs(MODEL_CACHE_DIR, mode=0o755, exist_ok=True)

			os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
			os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

			print("🔄 Loading tokenizer...")
			try:
				_tokenizer = AutoTokenizer.from_pretrained(
					MODEL_NAME,
					cache_dir=MODEL_CACHE_DIR,
					trust_remote_code=True,
					use_fast=True,
					force_download=False
				)
			except Exception as e:
				print(f"⚠️ Tokenizer loading failed: {e}")
				print("🔄 Trying alternative tokenizer loading...")
				_tokenizer = AutoTokenizer.from_pretrained(
					"microsoft/BioGPT",
					cache_dir=MODEL_CACHE_DIR,
					trust_remote_code=True
				)
			
			if _tokenizer.pad_token is None:
				_tokenizer.pad_token = _tokenizer.eos_token
			
			print("✅ Tokenizer loaded successfully!")
			
			print("🔄 Configuring 4-bit quantization...")
			bnb_config = BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_use_double_quant=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
			)
			
			print("🔄 Loading BioGPT model...")
			model = None
			device_info = "CPU"
			
			try:
				print("   • Attempting simple loading...")
				model = AutoModelForCausalLM.from_pretrained(
					MODEL_NAME,
					cache_dir=MODEL_CACHE_DIR,
					trust_remote_code=True,
					torch_dtype=torch.float32,
					low_cpu_mem_usage=True
				)
				device_info = "CPU (FP32)"
				print("   ✅ Simple loading successful!")
			except Exception as e:
				print(f"   ⚠️ Simple loading failed: {e}")
				
			if model is None:
				try:
					print("   • Attempting loading with float16...")
					model = AutoModelForCausalLM.from_pretrained(
						MODEL_NAME,
						cache_dir=MODEL_CACHE_DIR,
						trust_remote_code=True,
						torch_dtype=torch.float16,
					)
					device_info = "CPU (FP16)"
					print("   ✅ Float16 loading successful!")
				except Exception as e:
					print(f"   ⚠️ Float16 loading failed: {e}")
			
			if model is None:
				try:
					print("   • Attempting basic loading...")
					model = AutoModelForCausalLM.from_pretrained(
						MODEL_NAME,
						cache_dir=MODEL_CACHE_DIR,
						trust_remote_code=True
					)
					device_info = "CPU (Default)"
					print("   ✅ Basic loading successful!")
				except Exception as e:
					print(f"   ⚠️ All loading strategies failed: {e}")
					raise Exception(f"Failed to load model with all strategies: {e}")
			
			if model is not None and torch.cuda.is_available():
				try:
					print("   • Moving model to GPU...")
					model = model.cuda()
					device_info = device_info.replace("CPU", "GPU")
					print("   ✅ Model moved to GPU!")
				except Exception as e:
					print(f"   ⚠️ GPU transfer failed, staying on CPU: {e}")
			
			print("🔄 Creating text generation pipeline...")
			device = 0 if torch.cuda.is_available() and "GPU" in device_info else -1
			
			_model_pipeline = pipeline(
				"text-generation",
				model=model,
				tokenizer=_tokenizer,
				max_new_tokens=80,
				do_sample=True,
				temperature=0.3,
				top_p=0.9,
				pad_token_id=_tokenizer.eos_token_id,
				eos_token_id=_tokenizer.eos_token_id
			)

			
			print(f"✅ BioGPT model loaded successfully!")
			print(f"📱 Device: {device_info}")
			print(f"🧠 Model: {MODEL_NAME}")
			print(f"💾 Cache: {MODEL_CACHE_DIR}")
			
		except Exception as e:
			print(f"❌ Model loading failed: {e}")
			print("\n🔧 Troubleshooting suggestions:")
			print("   • Check internet connection")
			print("   • Ensure sufficient disk space (>2GB)")
			print("   • Try running with administrator privileges")
			print("   • Clear model cache and retry")
			print(f"   • Delete {MODEL_CACHE_DIR} folder and retry")
			raise


@lru_cache(maxsize=32)
def get_cached_response(condition_hash):
	"""Cache responses for common conditions to improve performance."""
	return None


def ask_chatdoctor(condition):
	"""Get triage recommendations using BioGPT with medical-specific prompting."""
	
	if _model_pipeline is None:
		preload_model()
	
	print("\n💬 ChatDoctor Response:\n")
	
	try:
		prompt = f"""<|system|>You are a medical triage assistant for Cleveland Clinic Abu Dhabi. Recommend one department based on symptoms.<|endoftext|>

<|user|>Patient symptoms: {condition}

Available departments: Emergency, Heart/Vascular/Thoracic, Dermatology, Digestive Diseases, Neurology, Ophthalmology, ENT, Pulmonary, Rheumatology, Endocrinology, Gynecology, Urology, Dentistry, Cancer, Psychiatry, Primary Care

Recommendation:<|endoftext|>

<|assistant|>Based on the symptoms, I recommend visiting"""

		response = _model_pipeline(
			prompt,
			max_new_tokens=50,
			do_sample=True,
			temperature=0.2,
			top_p=0.9,
			top_k=50,
			repetition_penalty=1.2,
			pad_token_id=_tokenizer.eos_token_id,
			eos_token_id=_tokenizer.eos_token_id,
			return_full_text=False
		)
		
		generated_text = response[0]['generated_text'].strip()
		
		recommendation = generated_text.split('\n')[0].strip()
		recommendation = recommendation.replace('<|endoftext|>', '').strip()
		
		if not recommendation or len(recommendation) < 10:
			recommendation = "Primary Care for initial evaluation and appropriate referral."
			
		final_response = f"Based on the symptoms, I recommend visiting {recommendation}"
		
		print(final_response)
		print()
		return final_response

	except Exception as e:
		error_msg = f"⚠️ ChatDoctor failed: {e}"
		print(error_msg)
		return "Sorry, I'm having technical difficulties. Please visit Primary Care or contact our support team."


class ChatDoctor:
	"""Optimized ChatDoctor class with conversation history for BioGPT."""
	
	def __init__(self):
		self.chat_history = []
		self.max_history = 3
		
	def chat(self, user_input):
		"""Interactive chat with conversation history using BioGPT."""
		
		if _model_pipeline is None:
			preload_model()
		
		self.chat_history.append(f"Patient: {user_input}")
		
		conversation_context = "\n".join(self.chat_history[-self.max_history:])
		
		print("\n💬 ChatDoctor:\n")
		
		try:
			prompt = f"""You are ChatDoctor, a clinical triage assistant for Cleveland Clinic Abu Dhabi.

Previous conversation:
{conversation_context}

Based on the patient's symptoms, recommend the most appropriate department from:
- Allergy & Immunology, Cancer, Dentistry, Dermatology, Digestive Diseases
- Endocrinology, Gynecology, Heart/Vascular/Thoracic, Nephrology, Neurology
- Ophthalmology, ENT, Pain Medicine, Primary Care, Psychiatry, Pulmonary
- Rheumatology, Urology, Emergency Department, Other

Provide a short, clear recommendation.

Doctor:"""

			response = _model_pipeline(
				prompt,
				max_new_tokens=80,
				do_sample=True,
				temperature=0.3,
				top_p=0.9,
				pad_token_id=_tokenizer.eos_token_id,
				eos_token_id=_tokenizer.eos_token_id
			)

			generated_text = response[0]['generated_text']
			doctor_start = generated_text.find("Doctor:") + len("Doctor:")
			if doctor_start > len("Doctor:") - 1:
				reply = generated_text[doctor_start:].strip()
			else:
				reply = generated_text[len(prompt):].strip()
			
			reply = reply.split('\n')[0].strip()
			if not reply:
				reply = "Please visit Primary Care for further evaluation."
			
			print(reply)
			print()
			
			self.chat_history.append(f"Doctor: {reply}")
			
			return reply
			
		except Exception as e:
			error_msg = f"⚠️ ChatDoctor failed: {e}"
			print(error_msg)
			return "Sorry, I'm having technical difficulties. Please try again."


def run_interactive_chat():
	"""Run interactive chat session with improved UX."""
	chatdoctor = ChatDoctor()
	
	print("🏥 Welcome to Cleveland Clinic Abu Dhabi ChatDoctor Triage Assistant")
	print("📞 To schedule your appointment, call our Contact Center (7:00am-7:00pm, Mon-Fri)")
	print("💬 Type 'quit' or 'exit' to end the session\n")
	
	while True:
		try:
			user_input = input("👤 Describe the patient's symptoms: ").strip()
			
			if not user_input:
				print("⚠️ Please describe the patient's condition so we can assist you.")
				continue
				
			if user_input.lower() in ['quit', 'exit', 'bye']:
				print("👋 Thank you for using ChatDoctor. Take care!")
				break
			
			chatdoctor.chat(user_input)
			
		except KeyboardInterrupt:
			print("\n👋 Session ended. Thank you for using ChatDoctor!")
			break
		except Exception as e:
			print(f"⚠️ Unexpected error: {e}")


if __name__ == "__main__":
	preload_model()
	
	if len(sys.argv) > 1:
		condition = " ".join(sys.argv[1:])
		ask_chatdoctor(condition)
	else:
		run_interactive_chat()
