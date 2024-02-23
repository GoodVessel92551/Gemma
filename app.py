# app.py (Flask backend)

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

access_token = "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"
model_id = "google/gemma-2b-it"
dtype = torch.bfloat16
current_answer = {"current_answer": "","end":False}

tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
    token=access_token,
)

chat_history = [{"role": "user", "content": ""},{"role": "assistant", "content": "Sure, I will use <img-prompt>to create images Only if the user ask for it."}]

@app.route('/')
def index():
    global chat_history
    chat_history = [{"role": "user", "content": "You are Booogle Chat, respond to the user as simple as possible."},{"role": "assistant", "content": "Sure"}]
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    global current_answer
    current_answer = {"current_answer": "","end":False}
    user_input = request.json['user_input']
    response = generate_response(user_input)
    return jsonify({'response': response})

def generate_response(user_input):
    global chat_history
    chat_history.append({"role": "user", "content": user_input})
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        while True:
            outputs = model.generate(input_ids=inputs, max_length=inputs.shape[1] + 1, num_return_sequences=1)
            generated_token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
            if generated_token == "":
                current_answer["end"] = True
                chat_history.append({"role": "assistant", "content": current_answer["current_answer"]})
                break
            elif generated_token == "img-prompt":
                print("Image Prompt")
                
            current_answer["current_answer"] += generated_token
            inputs = torch.cat([inputs, outputs[:, -1:].to("cuda")], dim=-1)  # Move

    return "Done"

@app.route("/get_current_answer")
def get_current_answer():
    return jsonify(current_answer)

if __name__ == '__main__':
    app.run(debug=True)