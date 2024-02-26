

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import torch
import base64
import os
from diffusers import AutoPipelineForText2Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
import gc
import scipy.io.wavfile
import io


image_url = ""

app = Flask(__name__)
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

access_token = "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"
model_id = "google/gemma-2b-it"
dtype = torch.bfloat16
current_answer = {"current_answer": "","end":False,"type":"chat"}

tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
    token=access_token,
)

image_prompts = ["create an image","create a image","create a photo","create a painting"]
audio_prompts = ["create a song","create a music","create a tune","create a melody"]

@app.route('/')
def index():
    global chat_history
    chat_history = [{"role": "user", "content": "You are Booogle Chat, respond to the user concise as possible. Your can chat with the user make images and songs when the users and"},{"role": "assistant", "content": "Sure"}]
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    global current_answer
    current_answer = {"current_answer": "","end":False,"type":"chat"}
    user_input = request.json['user_input']
    for i in image_prompts:
        if i in user_input.lower():
            print("Creating an image")
            generate_image(user_input)
            response = "Creating an image..."
            break
    else:
        for i in audio_prompts:
            if i in user_input.lower():
                print("Creating a song")
                generate_music(user_input)
                response = "Creating a song..."
                break
        else:
            global model_id, access_token, dtype,model,tokenizer
            try:
                model
            except:
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="cuda",
                    torch_dtype=dtype,
                    token=access_token,
                )
            response = generate_response(user_input)
    current_answer["end"] = True
    return jsonify({'response': response})

def generate_response(user_input):
    global chat_history
    chat_history.append({"role": "user", "content": user_input})
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to("cuda")

    generated_responses = []

    with torch.no_grad():
        while True:
            outputs = model.generate(input_ids=inputs, max_length=inputs.shape[1] + 1, num_return_sequences=1,temperature=1)  # Batch size of 5
            for output in outputs:
                generated_token = tokenizer.decode(output[-1:], skip_special_tokens=True)
                if generated_token == "":
                    current_answer["end"] = True
                    chat_history.append({"role": "assistant", "content": current_answer["current_answer"]})
                    current_answer["end"] = True
                    generated_responses.append(current_answer["current_answer"])
                    current_answer["type"] = "chat"
                    break
                
                current_answer["current_answer"] += generated_token
                inputs = torch.cat([inputs, outputs[:, -1:].to("cuda")], dim=-1)
            if current_answer["end"]:
                break

    return generated_responses

def generate_image(prompt):
    global model, tokenizer
    try:
        del model
        del tokenizer
        gc.collect()
    except:
        pass
    current_answer["current_answer"] = []
    for _ in range(2):
        output_image = pipe(prompt=prompt, num_inference_steps=3, guidance_scale=0.0).images[0]
        img_byte_array = io.BytesIO()
        output_image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()
        base64_image = base64.b64encode(img_byte_array).decode('utf-8')
        data_url = f"data:image/png;base64,{base64_image}"
        current_answer["current_answer"].append(data_url)
        print("Image created")
    current_answer["type"] = "image"
    current_answer["end"] = True
    return data_url

def generate_music(prompt):
    global model, tokenizer
    del model
    del tokenizer
    gc.collect()
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )

    audio_values = music_model.generate(**inputs, max_new_tokens=350)

    sampling_rate = music_model.config.audio_encoder.sampling_rate
    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, rate=sampling_rate, data=audio_values[0, 0].numpy())

    data_url = "data:audio/wav;base64," + base64.b64encode(wav_io.getvalue()).decode()

    current_answer["current_answer"] = data_url
    current_answer["type"] = "audio"
    current_answer["end"] = True
    global model_id, access_token, dtype
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=dtype,
        token=access_token,
    )
    return data_url


@app.route("/get_current_answer")
def get_current_answer():
    if current_answer["type"] == "image":
        return jsonify(current_answer)
    return jsonify(current_answer)

app.run(debug=True)
