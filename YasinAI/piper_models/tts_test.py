import warnings
warnings.filterwarnings("ignore")

from piper import PiperVoice
import wave

print("Loading voice model...")
voice = PiperVoice.load("/mnt/c/Users/yasin/Desktop/YasinAi/venv/piper_models/ar_JO-kareem-medium.onnx")

text = "تم إدخال المريض إلى قسم الطوارئ وهو يعاني من صداع حاد مستمر وفقدان مؤقت للوعي. بعد الفحص الأولي، لاحظ الفريق الطبي وجود ضعف جزئي في الجانب الأيمن من الجسم مع اضطراب في النطق، مما أثار الاشتباه في جلطة دماغية. تم استدعاء طبيب الأعصاب فورًا لإجراء تقييم دقيق للحالة، حيث أوصى بعمل تصوير بالرنين المغناطيسي (MRI) و تصوير مقطعي (CT Scan) للدماغ لتحديد موقع الإصابة. كما تم أخذ عينات لإجراء تحاليل الدم للتحقق من مستوى السكر والكوليسترول واحتمال وجود تجلط دموي. بعد ظهور النتائج، أكد طبيب الأعصاب أن الحالة تحتاج إلى علاج دوائي فوري بمذيبات الجلطات مع مراقبة مستمرة للعلامات الحيوية. بالإضافة إلى ذلك، تم تحويل المريض لاحقًا إلى قسم العلاج الطبيعي والتأهيل العصبي لاستعادة الحركة تدريجيًا وتحسين التوازن العصبي العضلي."
output_path = "test_output.wav"

print("Generating speech...")

# Set speed in the voice config
voice.config.length_scale = 0.75  # 25% faster

# Synthesize
audio_chunks = list(voice.synthesize(text))

# Write to WAV file
with wave.open(output_path, "wb") as wav_file:
    if audio_chunks:
        first_chunk = audio_chunks[0]
        wav_file.setnchannels(first_chunk.sample_channels)
        wav_file.setsampwidth(first_chunk.sample_width)
        wav_file.setframerate(first_chunk.sample_rate)
        
        # Write all audio data
        for chunk in audio_chunks:
            wav_file.writeframes(chunk.audio_int16_bytes)

print(f"✓ Audio saved to {output_path}")