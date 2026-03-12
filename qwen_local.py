import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("載入 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("載入 model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {"role": "system", "content": "你是一個簡潔、清楚的中文助理。"}
]

print("模型已就緒，輸入 exit 離開。")

while True:
    user_input = input("你: ").strip()
    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("Qwen:", reply)
    messages.append({"role": "assistant", "content": reply})