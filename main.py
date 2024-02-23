import tkinter as tk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore")

class ChatApp:
    def __init__(self, master):
        self.master = master
        master.title("Chat with Assistant")

        self.chat_history = tk.Text(master, state='disabled')
        self.chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.user_input = tk.Entry(master)
        self.user_input.grid(row=1, column=0, padx=10, pady=10)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        # Load model
        access_token = "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"
        model_id = "google/gemma-2b-it"
        dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=dtype,
            token=access_token,
        )

        self.chat_history_content = [{"role": "user", "content": "Use <e> as one token to end the message."},
                     {"role": "assistant", "content": "Sure, I will try to be as simple as possible."}]

    def send_message(self):
        user_prompt = self.user_input.get()
        self.user_input.delete(0, 'end')
        self.chat_history.configure(state='normal')
        self.chat_history.insert('end', f"You: {user_prompt}\n")

        self.generate_response(user_prompt)

        self.chat_history.insert('end', f"Assistant: {self.chat_history_content[-1]['content']}\n")
        self.chat_history.configure(state='disabled')
        self.chat_history.see('end')

    def generate_response(self, user_prompt):
        self.chat_history_content.append({"role": "user", "content": user_prompt})
        prompt = self.tokenizer.apply_chat_template(self.chat_history_content, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(self.model.device) 

        with torch.no_grad():
            for _ in range(200):  
                outputs = self.model.generate(input_ids=inputs, max_length=inputs.shape[1] + 1)
                generated_token = self.tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
                if generated_token == "":
                    break
                self.chat_history_content[-1]["content"] += generated_token
                inputs = torch.cat([inputs, outputs[:, -1:].to(self.model.device)], dim=-1)

        self.chat_history_content.append({"role": "assistant", "content": self.tokenizer.decode(outputs[0], skip_special_tokens=True)})


def main():
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
