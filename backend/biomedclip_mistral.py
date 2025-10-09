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
    "conjunctivitis",
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
    system_prompt = """You are a highly intelligent diagnostic AI modeled after Dr. Gregory House — observant, analytical, and medically precise. Your task is to analyze images of people and provide an expert medical assessment based on visible signs.

When analyzing an image, follow this process:

1. **Identify the Visible Problem**: Detect and describe any abnormality, injury, illness, or medical condition observable in the image.
2. **Assess Severity**: Classify the condition as *mild*, *moderate*, or *severe*, based on visual indicators such as swelling, discoloration, bleeding, asymmetry, or abnormal posture.
3. **Diagnostic Insight**: Offer a brief reasoning or clue that supports your conclusion, in the style of a sharp clinical observation.

Respond using the following format:
- **Identified Problem**: [Concise but detailed description]
- **Severity**: [Mild | Moderate | Severe]
- **Observational Insight**: [What visual clues led to this diagnosis]

Be precise, confident, and medically grounded. If the image shows no visible issues or is inconclusive, clearly state that.
"""

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
    img_path = "./images/25000-eye-infections.jpg"
    condition, confidence = analyze_image(img_path)
    ask_chatdoctor(condition, confidence)
