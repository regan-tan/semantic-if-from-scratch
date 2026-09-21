from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

examples = [
    "A",
    "B",
    "C",
    "yes",
    "no",
    "Billing",
    "Technical Support",
]

for text in examples:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"Text: {text!r}")
    print(f"Token IDs: {token_ids}")
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(token_ids)}")
    print("-" * 50)