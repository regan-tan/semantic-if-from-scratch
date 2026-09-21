# Semantic If From Scratch

A learning project that explores how language models can be used as **semantic decision engines** instead of only as text generators.

The goal is to understand the technical ideas behind systems such as OpenJEV / SemIf by rebuilding the core mechanism step by step using PyTorch, Hugging Face Transformers, and Qwen3-0.6B.

## How It Works

A causal language model predicts a logit for every possible next token in its vocabulary.

Instead of letting the model generate a full response, this project directly reads the logits corresponding to predefined decision tokens such as:

- `A = Yes`
- `B = No`

The selected logits are normalized with softmax and then used to make a semantic decision.

```text
Input text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Qwen3-0.6B
    ↓
Next-token logits
    ↓
Select option logits
    ↓
Restricted softmax
    ↓
Semantic decision
```

For example:

```text
Context:
"I was charged twice for my subscription."

Condition:
"The customer's message concerns billing."

A = Yes
B = No
```

The model is run once to obtain its next-token logits. The program then compares only the logits for `A` and `B`, instead of generating a textual answer.

## Current Progress

The project currently covers:

- Tokenization and token IDs
- Normal autoregressive LLM generation
- Inspection of raw next-token logits
- Direct A/B/C option scoring
- Restricted softmax over selected options
- A reusable `semantic_if()` function
- Basic automated tests
- Experiments for evaluating model behaviour on semantic decisions

The project is still experimental. Restricted-softmax probabilities should not be interpreted as calibrated real-world confidence, and model behaviour can be affected by prompt wording, option labels, and model capability.

## Project Structure

```text
semantic-if-from-scratch/
│
├── src/
│   ├── 01_tokenizer.py
│   ├── 02_generate.py
│   ├── 03_logits.py
│   ├── 04_direct_decision.py
│   ├── 05_semantic_if.py
│   └── semantic_if.py
│
├── experiments/
│   └── semantic_if_cases.py
│
├── tests/
│   └── test_semantic_if.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

The numbered files in `src/` document the learning process step by step, while `src/semantic_if.py` contains the reusable implementation.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the tokenizer experiment:

```bash
python src/01_tokenizer.py
```

Run normal LLM generation:

```bash
python src/02_generate.py
```

Inspect next-token logits:

```bash
python src/03_logits.py
```

Run direct option scoring:

```bash
python src/04_direct_decision.py
```

Run the semantic-if example:

```bash
python src/05_semantic_if.py
```

Run experiments:

```bash
python experiments/semantic_if_cases.py
```

Run automated tests:

```bash
pytest -v
```

## Model

This project currently uses:

```text
Qwen/Qwen3-0.6B
```

The model runs locally using PyTorch.

## Key Idea

Normal LLM generation:

```text
Prompt
  ↓
Model
  ↓
Generate token
  ↓
Generate another token
  ↓
Generate another token
  ↓
Parse final response
```

Direct semantic decision:

```text
Prompt
  ↓
Model
  ↓
Next-token logits
  ↓
Read only the option logits
  ↓
Make decision
```

This allows the language model to act as a semantic component inside normal program control flow.

## License

MIT License.