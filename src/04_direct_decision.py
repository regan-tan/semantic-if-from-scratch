import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"


# --------------------------------------------------
# 1. Load tokenizer and model
# --------------------------------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

model.eval()


# --------------------------------------------------
# 2. Define the decision problem
# --------------------------------------------------

options = {
    "A": "Billing",
    "B": "Technical Support",
    "C": "Sales",
}


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


# --------------------------------------------------
# 3. Format and tokenize the prompt
# --------------------------------------------------

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


# --------------------------------------------------
# 4. Run ONE forward pass
# --------------------------------------------------

with torch.no_grad():
    outputs = model(**inputs)


# --------------------------------------------------
# 5. Get logits for the NEXT token only
# --------------------------------------------------

next_token_logits = outputs.logits[0, -1, :].float()


# --------------------------------------------------
# 6. Find the token ID for A, B and C
# --------------------------------------------------

option_token_ids = {}

for label in options:
    token_ids = tokenizer.encode(
        label,
        add_special_tokens=False,
    )

    if len(token_ids) != 1:
        raise ValueError(
            f"Option label {label!r} is not one token: {token_ids}"
        )

    option_token_ids[label] = token_ids[0]


print("\nOPTION TOKEN IDS:")

for label, token_id in option_token_ids.items():
    print(
        f"{label} -> token ID {token_id}"
    )


# --------------------------------------------------
# 7. Extract ONLY the logits for A, B and C
# --------------------------------------------------

selected_logits = torch.stack(
    [
        next_token_logits[option_token_ids[label]]
        for label in options
    ]
)


print("\nRAW OPTION LOGITS:")

for label, logit in zip(
    options,
    selected_logits,
):
    print(
        f"{label} ({options[label]}): "
        f"{logit.item():.4f}"
    )


# --------------------------------------------------
# 8. Softmax ONLY across A, B and C
# --------------------------------------------------

selected_probabilities = torch.softmax(
    selected_logits,
    dim=0,
)


print("\nRESTRICTED OPTION PROBABILITIES:")

for label, probability in zip(
    options,
    selected_probabilities,
):
    print(
        f"{label} ({options[label]}): "
        f"{probability.item():.4%}"
    )


# --------------------------------------------------
# 9. Pick the highest-scoring option
# --------------------------------------------------

winner_index = torch.argmax(
    selected_probabilities
).item()

labels = list(options.keys())

winner_label = labels[winner_index]
winner_option = options[winner_label]


print("\nDECISION:")
print(
    f"{winner_label} -> {winner_option}"
)