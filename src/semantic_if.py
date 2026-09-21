import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

model.eval()


def semantic_if(
    context: str,
    condition: str,
):
    """
    Evaluate whether a semantic condition is true or false
    using direct next-token logits.

    A = Yes
    B = No
    """

    messages = [
        {
            "role": "user",
            "content": f"""
Given the following context:

{context}

Evaluate this condition:

{condition}

Options:
A. Yes
B. No

Return only A or B.
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

    next_token_logits = outputs.logits[
        0,
        -1,
        :
    ].float()

    labels = ["A", "B"]

    option_token_ids = {}

    for label in labels:
        token_ids = tokenizer.encode(
            label,
            add_special_tokens=False,
        )

        if len(token_ids) != 1:
            raise ValueError(
                f"Option label {label!r} is not exactly "
                f"one token: {token_ids}"
            )

        option_token_ids[label] = token_ids[0]

    selected_logits = torch.stack(
        [
            next_token_logits[
                option_token_ids[label]
            ]
            for label in labels
        ]
    )

    selected_probabilities = torch.softmax(
        selected_logits,
        dim=0,
    )

    yes_probability = selected_probabilities[0].item()
    no_probability = selected_probabilities[1].item()

    decision = yes_probability > no_probability

    return {
        "yes": yes_probability,
        "no": no_probability,
        "decision": decision,
        "raw_logits": {
            "A": selected_logits[0].item(),
            "B": selected_logits[1].item(),
        },
        "token_ids": {
            "A": option_token_ids["A"],
            "B": option_token_ids["B"],
        },
    }