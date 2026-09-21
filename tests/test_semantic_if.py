import sys
from pathlib import Path

import pytest


# --------------------------------------------------
# Allow tests/ to import code from src/
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from semantic_if import semantic_if


# --------------------------------------------------
# Test 1: Clear YES case
# --------------------------------------------------

def test_obvious_billing_message_returns_true():
    result = semantic_if(
        context="""
I was charged twice for my subscription
and I want a refund immediately.
""",
        condition="""
The customer's message concerns billing.
""",
    )

    assert result["decision"] is True
    assert result["yes"] > result["no"]


# --------------------------------------------------
# Test 2: Clear NO case
# --------------------------------------------------

def test_obvious_non_billing_message_returns_false():
    result = semantic_if(
        context="""
The application crashes every time
I try to upload a photo.
""",
        condition="""
The customer's message concerns billing.
""",
    )

    assert result["decision"] is False
    assert result["no"] > result["yes"]


# --------------------------------------------------
# Test 3: Probabilities should sum to approximately 1
# --------------------------------------------------

def test_probabilities_sum_to_one():
    result = semantic_if(
        context="""
I was charged twice for my subscription.
""",
        condition="""
The customer's message concerns billing.
""",
    )

    total_probability = (
        result["yes"]
        + result["no"]
    )

    assert total_probability == pytest.approx(
        1.0,
        abs=1e-6,
    )


# --------------------------------------------------
# Test 4: Probabilities must be between 0 and 1
# --------------------------------------------------

def test_probabilities_are_valid():
    result = semantic_if(
        context="""
I need help with my account.
""",
        condition="""
The customer's message concerns billing.
""",
    )

    assert 0.0 <= result["yes"] <= 1.0
    assert 0.0 <= result["no"] <= 1.0


# --------------------------------------------------
# Test 5: Decision should always be a Boolean
# --------------------------------------------------

def test_decision_is_boolean():
    result = semantic_if(
        context="""
Something seems wrong with my account.
""",
        condition="""
The customer's message concerns billing.
""",
    )

    assert isinstance(
        result["decision"],
        bool,
    )


# --------------------------------------------------
# Test 6: A and B must map to different token IDs
# --------------------------------------------------

def test_option_token_ids_are_different():
    result = semantic_if(
        context="""
I was charged twice.
""",
        condition="""
The customer's message concerns billing.
""",
    )

    assert (
        result["token_ids"]["A"]
        != result["token_ids"]["B"]
    )


# --------------------------------------------------
# Test 7: Raw logits should be numeric values
# --------------------------------------------------

def test_raw_logits_are_numbers():
    result = semantic_if(
        context="""
I was charged twice.
""",
        condition="""
The customer's message concerns billing.
""",
    )

    assert isinstance(
        result["raw_logits"]["A"],
        float,
    )

    assert isinstance(
        result["raw_logits"]["B"],
        float,
    )