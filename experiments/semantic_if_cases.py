import sys
from pathlib import Path


# Allow this experiment file to import from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from semantic_if import semantic_if


test_cases = [
    {
        "name": "Obvious yes",
        "context": """
I was charged twice for my subscription
and I want a refund immediately.
""",
        "condition": """
The customer's message concerns billing.
""",
    },
    {
        "name": "Obvious no",
        "context": """
The app crashes every time I try to upload a photo.
""",
        "condition": """
The customer's message concerns billing.
""",
    },
    {
        "name": "Ambiguous",
        "context": """
Something seems wrong with my account.
""",
        "condition": """
The customer's message concerns billing.
""",
    },
    {
        "name": "Missing information",
        "context": """
Hello, I need some help.
""",
        "condition": """
The customer's message concerns billing.
""",
    },
    {
        "name": "Mixed issue",
        "context": """
I was charged for my subscription,
but the application also keeps crashing.
""",
        "condition": """
The customer's message concerns billing.
""",
    },
]


for test_case in test_cases:

    result = semantic_if(
        context=test_case["context"],
        condition=test_case["condition"],
    )

    print("\n" + "=" * 60)

    print("TEST:")
    print(test_case["name"])

    print("\nCONTEXT:")
    print(test_case["context"].strip())

    print("\nCONDITION:")
    print(test_case["condition"].strip())

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

    print("\nDECISION:")
    print(result["decision"])