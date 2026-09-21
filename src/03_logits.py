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


inputs = tokenizer(
    text,
    return_tensors="pt",
)


with torch.no_grad():
    outputs = model(**inputs)


print("\nFULL LOGITS SHAPE:")
print(outputs.logits.shape)


next_token_logits = outputs.logits[0, -1, :]


print("\nNEXT TOKEN LOGITS SHAPE:")
print(next_token_logits.shape)


probabilities = torch.softmax(
    next_token_logits,
    dim=-1,
)


top_probs, top_ids = torch.topk(
    probabilities,
    k=10,
)


print("\nTOP 10 NEXT-TOKEN PREDICTIONS:")

for probability, token_id in zip(
    top_probs,
    top_ids,
):
    token_id = token_id.item()

    token = tokenizer.convert_ids_to_tokens(
        token_id
    )

    decoded = tokenizer.decode(
        [token_id]
    )

    print(
        f"ID: {token_id:<8} "
        f"Token: {token!r:<20} "
        f"Decoded: {decoded!r:<15} "
        f"Probability: {probability.item():.6f}"
    )