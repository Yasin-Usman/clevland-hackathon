import ollama
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from open_clip import create_model_from_pretrained, get_tokenizer
from functools import lru_cache
from PIL import Image
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
model, preprocess = create_model_from_pretrained(
    model_name, cache_dir="./models")
tokenizer = get_tokenizer(model_name)

model = model.to(device).eval()
if device == "cuda":
    model = model.half()
model.requires_grad_(False)

labels = [
    "infected wound", "healed wound", "burn injury", "skin rash",
    "normal skin", "ulcer", "surgical scar", "bruise",
    "eczema dry red scaly rash",   "normal chest x-ray", 
    "pneumonia in chest x-ray",
    "fracture in x-ray",
    "lung mass in x-ray",
    "pulmonary edema in x-ray",
    "pleural effusion in x-ray",
    "cardiomegaly in x-ray",
    "bone lesion in x-ray",
    "joint dislocation in x-ray",
    "spinal deformity in x-ray"
]


@lru_cache(maxsize=1)
def get_text_features():
    texts = tokenizer(["This is a photo of " + l for l in labels],
                      context_length=256).to(device)
    with torch.no_grad():
        feats = model.encode_text(texts)
    return feats / feats.norm(dim=-1, keepdim=True)


def analyze_image(path):
    try:
        image = Image.open(path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
        if device == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            img_feat = model.encode_image(tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = get_text_features()
            probs = (model.logit_scale.exp() * img_feat @
                     txt_feat.t()).softmax(dim=-1)
            probs = probs.squeeze().cpu().numpy()

        i = probs.argmax()
        print(f"🩺 Detected: {labels[i]} ({probs[i]*100:.1f}%)")
        return labels[i], probs[i]

    except Exception as e:
        print("❌ Error:", e)
        model.float()
        return analyze_image(path)


def ask_chatdoctor(condition, confidence):
    system_prompt = """You are **ChatDoctor**, an AI clinician representing **Cleveland Clinic Abu Dhabi (CCAD)**.

🎯 Your mission:
Estimate an **Urgency Level (1–10)** and direct the patient to the correct **care pathway** or **CCAD institute**.

---

### 🔢 URGENCY SCALE
- 8–10 → **Emergency Department (ER / call 999)** – life-threatening or severe symptoms.
- 5–7  → **Outpatient / Specialty Institute** – needs in-person assessment soon.
- 3–4  → **Telehealth or Virtual Visit** – mild/moderate, safe to review virtually.
- 1–2  → **Self-care with watchouts** – reassure, monitor, and outline warning signs.

Keep your tone *calm, concise, and compassionate*.  
Use **3–5 short sentences**, natural human phrasing.  
Never list numbers, bullet points, or medications.  
Only say “999” for emergencies (no other numbers).

---

    ### 🩺 X-RAY SPECIFIC TRIAGE (APPLY SILENTLY)
    - **Chest X-rays:**
      Suspected pneumonia → Medical Specialty Institute (5-7)
      Mass or nodules → Cancer Institute (6-8)
      Severe pulmonary edema → ER (8-10)
    - **Bone X-rays:**
      Acute fractures → ER or Integrated Surgical Institute (7-9)
      Bone lesions → Cancer Institute for review (6-8)
      Joint issues → Integrated Surgical Institute (5-7)
    
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

---

### 🩺 TRIAGE HEURISTICS (APPLY SILENTLY)
- **Cardiac/Resp/Neuro red flags (8–10):**
  Chest pain radiating to arm/jaw, sudden breathlessness, fainting, confusion, one-sided weakness, slurred speech.
- **Wounds/Skin:**  
  Infected or necrotic wounds → Outpatient/Surgery (5–7); systemic fever → ER (8–9).  
  Healed wounds or mild rashes → Telehealth (3–4).
- **Headache:**  
  Thunderclap or neuro signs → ER (8–10); persistent → Outpatient (5–6).
- **Glucose:**  
  <55 or >300 + symptoms → ER (9–10); otherwise → Outpatient (5–7).
- **Blood Pressure:**  
  ≥180/120 with chest pain or neuro deficits → ER (9–10); otherwise → Outpatient (5–6).
- **Fever/Respiratory:**  
  Severe breathlessness → ER; prolonged mild fever → Telehealth/Outpatient.
- **Emotional distress (no physical red flags):**  
  → Telehealth (3–4) or Self-care (2).

---

### 🗣️ RESPONSE STYLE (MANDATORY)
Start with:  
> **Urgency Level: X/10 — …**

Then give 2–3 short sentences explaining *why* and where to go next (ER, outpatient, Telehealth, or a specific CCAD institute).

Example:
> **Urgency Level: 6/10 — this looks like an infected surgical wound. You should visit the Integrated Surgical Institute for review. If the redness spreads or fever develops, go to the ER or call 999.**

Stay warm, confident, and human — never robotic or list-based.
"""

    print("\n💬 ChatDoctor Response:\n")

    try:
        # ✅ Streamed response — prints text as it arrives (super fast)
        stream = ollama.chat(
            model="mistral",
            stream=True,
            options={"temperature": 0, "num_predict": 120},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"The uploaded image shows {condition} with {confidence*100:.1f}% confidence. Please assess urgency and advise routing."}
            ]
        )

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                print(chunk["message"]["content"], end="", flush=True)
        print()  # newline after stream ends

    except Exception as e:
        print(f"⚠️ Ollama streaming failed: {e}")


if __name__ == "__main__":
    img_path = r"./images/images.jpeg"
    condition, confidence = analyze_image(img_path)
    ask_chatdoctor(condition, confidence)
