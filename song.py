from transformers import AutoProcessor, MusicgenForConditionalGeneration
import base64

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["cheerful tune reminiscent of a sunny day in a bustling city"],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=512)

# Convert audio data to base64
audio_base64 = base64.b64encode(audio_values[0, 0].numpy()).decode('utf-8')

# Create data URL
data_url = f"data:audio/wav;base64,{audio_base64}"