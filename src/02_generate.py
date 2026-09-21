import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

model.eval()


messages = [
    {
        "role": "user",
        "content": """
Classify this customer message.

Customer:
"I was charged twice for my subscription."

Options:
A. Billing
B. Technical Support
C. Sales

Return only A, B, or C.
""",
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)


print("\nPROMPT ACTUALLY SENT TO MODEL:")
print(text)


inputs = tokenizer(
    text,
    return_tensors="pt",
)


print("\nINPUT TENSOR SHAPE:")
print(inputs["input_ids"].shape)


with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
    )


new_tokens = generated[0][inputs["input_ids"].shape[1]:]


print("\nGENERATED TOKEN IDS:")
print(new_tokens)


print("\nGENERATED TOKENS:")
print(
    tokenizer.convert_ids_to_tokens(
        new_tokens.tolist()
    )
)


answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True,
)


print("\nMODEL ANSWER:")
print(repr(answer))