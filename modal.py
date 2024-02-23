from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")

access_token = "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"
model_id = "google/gemma-2b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
    token=access_token,
)

chat = []
chat.append({"role": "user", "content": "Use <e> as one token to end the message."})
chat.append({"role": "assistant", "content": "Sure, I will try to be as simple as possible."})

def chat(user_prompt):
    chat.append({"role": "user", "content": user_prompt})
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to("cuda")  # Move inputs to device

    # Generate one token at a time
    with torch.no_grad():
        for _ in range(200):  # Generate one token at a time
            outputs = model.generate(input_ids=inputs, max_length=inputs.shape[1] + 1)
            generated_token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
            if generated_token == "":
                break
            print(generated_token, end='', flush=True)  # Print the token without newline
            inputs = torch.cat([inputs, outputs[:, -1:].to("cuda")], dim=-1)  # Move outputs to device and append the latest token to inputs

    print("\n")
    chat.append({"role": "assistant", "content": tokenizer.decode(outputs[0], skip_special_tokens=True)})
