import ollama

async def stream_chatdoctor(condition: str):
    """Async genif __name__ == "__main__":
    import asyncio
    
    async def main():
        user_input = input("🩺 Describe the patient's symptoms: ")
        async for chunk in stream_chatdoctor(user_input):
            print(chunk, end="", flush=True)
        print()

    asyncio.run(main())tor that yields chunks of the response."""
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
        # Streamed Mistral response
        stream = ollama.chat(
            model="mistral",
            stream=True,
            options={"temperature": 0, "num_predict": 120},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Patient reports: {condition}. Please assess urgency and advise routing."}
            ]
        )

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                piece = chunk["message"]["content"]
                print(piece, end="", flush=True)  # Debug output
                yield piece  # Stream to client

        print()  # Debug newline after completion

    except Exception as e:
        error_msg = f"⚠️ Ollama streaming failed: {e}"
        print(error_msg)  # Debug output
        yield error_msg  # Send error to client


if __name__ == "__main__":
    user_input = input("🩺 Describe the patient’s symptoms: ")
    ask_chatdoctor(user_input)
