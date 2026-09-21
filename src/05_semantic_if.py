import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"


# --------------------------------------------------
# 1. Load tokenizer and model once
# --------------------------------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

model.eval()


# --------------------------------------------------
# 2. Reusable semantic_if function
# --------------------------------------------------

def semantic_if(
    context: str,
    condition: str,
):
    """
    Evaluate whether a semantic condition is true or false.

    The LLM is asked to choose between:

    A. Yes
    B. No

    We do NOT generate text.

    Instead, we inspect the model's next-token logits
    for A and B directly, then apply softmax only
    across those two options.
    """

    # ----------------------------------------------
    # Build the prompt
    # ----------------------------------------------

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

    # ----------------------------------------------
    # Convert chat messages into Qwen's expected
    # prompt format
    # ----------------------------------------------

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # ----------------------------------------------
    # Convert the formatted prompt into token IDs
    # ----------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    # ----------------------------------------------
    # Run ONE forward pass through the model
    # ----------------------------------------------

    with torch.no_grad():
        outputs = model(**inputs)

    # ----------------------------------------------
    # Get logits for the NEXT token only
    #
    # Shape before:
    # [batch_size, sequence_length, vocabulary_size]
    #
    # After:
    # [vocabulary_size]
    # ----------------------------------------------

    next_token_logits = outputs.logits[
        0,
        -1,
        :
    ].float()

    # ----------------------------------------------
    # Our allowed decision labels
    # ----------------------------------------------

    labels = ["A", "B"]

    option_token_ids = {}

    # ----------------------------------------------
    # Find token IDs for A and B
    # ----------------------------------------------

    for label in labels:
        token_ids = tokenizer.encode(
            label,
            add_special_tokens=False,
        )

        if len(token_ids) != 1:
            raise ValueError(
                f"Option label {label!r} is not exactly one token: "
                f"{token_ids}"
            )

        option_token_ids[label] = token_ids[0]

    # ----------------------------------------------
    # Extract ONLY the logits for A and B
    # ----------------------------------------------

    selected_logits = torch.stack(
        [
            next_token_logits[
                option_token_ids[label]
            ]
            for label in labels
        ]
    )

    # ----------------------------------------------
    # Softmax ONLY across A and B
    #
    # This gives relative probabilities:
    #
    # P(A | A or B)
    # P(B | A or B)
    # ----------------------------------------------

    selected_probabilities = torch.softmax(
        selected_logits,
        dim=0,
    )

    yes_probability = selected_probabilities[0].item()
    no_probability = selected_probabilities[1].item()

    # ----------------------------------------------
    # Final Boolean decision
    # ----------------------------------------------

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


# --------------------------------------------------
# 3. Example usage
# --------------------------------------------------

message = """
I was charged twice for my subscription
and I want a refund immediately.
"""


condition = """
The customer's message concerns billing.
"""


result = semantic_if(
    context=message,
    condition=condition,
)


# --------------------------------------------------
# 4. Print results
# --------------------------------------------------

print("\nCONTEXT:")
print(message.strip())

print("\nCONDITION:")
print(condition.strip())

print("\nTOKEN IDS:")
print(
    f"A -> {result['token_ids']['A']}"
)
print(
    f"B -> {result['token_ids']['B']}"
)

print("\nRAW LOGITS:")
print(
    f"A (Yes): {result['raw_logits']['A']:.4f}"
)
print(
    f"B (No):  {result['raw_logits']['B']:.4f}"
)

print("\nRESTRICTED PROBABILITIES:")
print(
    f"Yes: {result['yes']:.4%}"
)
print(
    f"No:  {result['no']:.4%}"
)

print("\nSEMANTIC DECISION:")
print(result["decision"])


# --------------------------------------------------
# 5. Example of using it like an if statement
# --------------------------------------------------

if result["decision"]:
    print(
        "\nAction: Route this message "
        "to the billing workflow."
    )
else:
    print(
        "\nAction: Do not route this message "
        "to the billing workflow."
    )