# Cleveland - Medical Image Analysis and Chat System

This repository contains two main applications:
1. A medical image analysis system using BiomedCLIP and Mistral AI
2. A text-based medical chat interface using Mistral AI

^

## Prerequisites^

^

- Python 3.8 or higher
- Ollama installed on your system ([Install Ollama](https://ollama.ai/))
- CUDA-capable GPU recommended (will fall back to CPU if unavailable)

^

## Installation^

^

1. Clone the repository:^

```bash^

git clone https://github.com/toxicMango64/cleveland.git^

cd cleveland^

```^

^

2. Install required Python packages:^

```bash^

pip install ollama
pip install torch
pip install Pillow
pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib huggingface_hub pillow
```^

^

3. Pull the required Mistral models:^

```bash^

# Pull the required Mistral model
ollama pull mistral

```^

^

## Usage^

^

### Text-Only Chat (mistral.py)^

^

1. Navigate to the backend directory:^

```bash^

cd backend^

```^

^

2. Run the text chat application:^

```bash^

python mistral.py^

```^

^

3. Start chatting with the AI. Type 'exit' to end the conversation.^

^

### Image Analysis (image_mistral.py)^

^

1. Navigate to the backend directory:^

```bash^

cd backend^

```^

^

2. Place your medical images in the `images` directory for analysis. The system can detect:
   - Infected wounds
   - Healed wounds
   - Burn injuries
   - Skin rashes
   - Normal skin
   - Ulcers
   - Surgical scars
   - Bruises
   - Eczema

3. Run the image analysis application:
```bash
python image_mistral.py
```

4. The system will:
   - Analyze the image using BiomedCLIP
   - Provide a diagnosis with confidence level
   - Use ChatDoctor (powered by Mistral) to:
     - Assess urgency (1-10 scale)
     - Recommend appropriate care pathway
     - Direct to specific Cleveland Clinic Abu Dhabi institute if needed

^

## Directory Structure^

^

```
backend/
├── image_mistral.py    # Medical image analysis implementation
├── mistral.py         # Medical chat implementation
├── images/           # Directory for image analysis
│   ├── e.png
│   ├── ech.jpg
│   └── mouth.jpeg
└── models/          # Directory for cached model files
```

## Technical Details

The image analysis system uses:
- Microsoft's BiomedCLIP model for medical image analysis
- Mistral AI for medical consultation
- CUDA acceleration when available (falls back to CPU)
- Caching for efficient model loading

^

## Notes^

^

- Ensure Ollama is running in the background before using either application
- For image analysis, place your medical images in the `backend/images` directory
- First-time use will download the BiomedCLIP model (approximately 1GB)
- GPU is recommended but not required
- The system provides medical guidance but should not replace professional medical consultation

^

## License^

^

This project is licensed under the terms of the LICENSE file included in the repository.^

^

## Contributing^

^

Feel free to submit issues and pull requests to improve the applications. > 