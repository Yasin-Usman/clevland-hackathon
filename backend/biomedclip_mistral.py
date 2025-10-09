import os
import io
import sys
import torch
import ollama
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from PIL import Image
from functools import lru_cache
from open_clip import create_model_from_pretrained, get_tokenizer

os.environ['HF_HUB_DISABLE_XET'] = '1'

warnings.filterwarnings("ignore", category=FutureWarning)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
# model_name = "./micrbosoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
cache_dir = os.path.expanduser("~/.cache/biomedclip")
os.makedirs(cache_dir, exist_ok=True)
model, preprocess = create_model_from_pretrained(
    model_name, cache_dir=cache_dir)
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
    "spinal deformity in x-ray",

    # Wound Types
    "open wound",
    "closed wound",
    "infected wound",
    "healed wound",
    "surgical wound",
    "bite wound",
    "crush injury",
    "skin tear",
    "abrasion",
    "laceration",
    "puncture wound",
    "incision",
    "dehisced wound",
    "traumatic wound",

    # Burns
    "burn",
    "first-degree burn",
    "second-degree burn",
    "third-degree burn",

    # Healing Features
    "granulation tissue",
    "epithelialization",
    "scab",
    "slough",
    "eschar",
    "necrosis",
    "maceration",
    "dry wound bed",
    "moist wound bed",
    "hypergranulation",

    # Discharge & Inflammation
    "purulent discharge",
    "serous drainage",
    "serosanguineous fluid",
    "bloody discharge",
    "pus",
    "erythema",
    "swelling",
    "edema",
    "foul odor",
    "skin inflammation",

    # Lesions
    "ulcer",
    "erosion",
    "papule",
    "pustule",
    "vesicle",
    "nodule",
    "plaque",
    "macule",
    "patch",
    "bullae",
    "wheal",
    "crust",
    "scale",
    "excoriation",
    "lichenification",
    "scar",
    "atrophic scar",
    "hypertrophic scar",
    "keloid",

    # Infections
    "abscess",
    "cellulitis",
    "impetigo",
    "folliculitis",
    "fungal infection",
    "candidiasis",
    "tinea corporis",
    "tinea pedis",
    "herpes simplex",
    "herpes zoster",
    "molluscum contagiosum",
    "warts",
    "scabies",

    # Chronic Wounds
    "diabetic ulcer",
    "venous ulcer",
    "arterial ulcer",
    "pressure ulcer",
    "stage 1 pressure ulcer",
    "stage 2 pressure ulcer",
    "stage 3 pressure ulcer",
    "stage 4 pressure ulcer",
    "unstageable ulcer",
    "deep tissue injury",

    # Dermatitis & Skin Conditions
    "eczema",
    "atopic dermatitis",
    "contact dermatitis",
    "seborrheic dermatitis",
    "psoriasis",
    "rosacea",
    "urticaria",
    "acne",
    "hidradenitis suppurativa",
    "chilblains",
    "ichthyosis",

    # Pigment & Texture Disorders
    "hyperpigmentation",
    "hypopigmentation",
    "discolored skin",
    "shiny skin",
    "peeling skin",
    "dry skin",
    "flaky skin",
    "thickened skin",
    "intact skin",
    "inflamed area",
    "fibrotic tissue",

    # Vascular Findings
    "cyanosis",
    "petechiae",
    "purpura",
    "ecchymosis",
    "bruise",
    "hematoma",
    "telangiectasia",
    "spider veins",
    "varicose veins",

    # Oncology & Growths
    "melanoma",
    "basal cell carcinoma",
    "squamous cell carcinoma",
    "actinic keratosis",
    "seborrheic keratosis",
    "dysplastic nevus",
    "nevus",
    "lentigo",
    "benign mole",
    "skin tag",
    "cyst",

    # Post-surgical & Treatment Sites
    "surgical scar",
    "skin graft",
    "donor site",
    "flap site",
    "stapled incision",
    "sutures",
    "drain site",

    # Medical Devices & Dressings
    "wound dressing",
    "negative pressure therapy",
    "surgical drain",
    "compression bandage",
    "catheter site"
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
    system_prompt = """You are ChatDoctor, a clinical triage assistant for Cleveland Clinic Abu Dhabi (CCAD).

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

Keep responses natural, concise, and focused. Do not list options or explain treatments. Never use technical jargon or urgency scores."""

    print("\n💬 ChatDoctor Response:\n")

    try:
        stream = ollama.chat(
            model="mistral",
            stream=True,
            options={"temperature": 1, "num_predict": 120},
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
    img_path = "./images/e.png"
    condition, confidence = analyze_image(img_path)
    ask_chatdoctor(condition, confidence)
