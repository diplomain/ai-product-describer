# AI Product Describer — Tiny GPT from Scratch

A small educational CLI that trains a **decoder-only GPT-style Transformer from scratch** on a starter product-description corpus, then generates new text from product details. The project implements byte-level BPE, token/position embeddings, masked causal self-attention, Transformer blocks, a training loop, and autoregressive sampling in PyTorch.

> This is a learning/demo model, not a commercial copywriter model. A tiny dataset and short CPU training run will produce rough, sometimes repetitive text. For better results, add a larger original or licensed corpus and train longer.

## Milestones implemented
- [x] **1. Corpus** — starter product descriptions saved as JSONL; customizable via `data/products.jsonl`.
- [x] **2. Tokenization & embeddings** — byte-level BPE with UTF-8-safe encoding/decoding; learned token and position embeddings.
- [x] **3. Transformer** — masked causal multi-head attention, residual connections, layer norm and feed-forward network.
- [x] **4. Training** — next-token prediction, cross entropy, AdamW, gradient clipping, model checkpoint save.
- [x] **5. Generation** — autoregressive sampling with temperature and top-k.
- [x] **6. CLI** — corpus, train, and generate commands.

## Requirements
- Python 3.10+
- PyTorch (CPU is enough; GPU optional)


## Windows PowerShell setup
```powershell
cd ai-product-describer
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.12 is not installed, use `py -m venv .venv` instead.

## Quick start
```powershell
# 1) Write the starter corpus (also happens automatically during training)
python -m app.cli data

# 2) Train a small local model (CPU; quick demo)
python -m app.cli train --steps 300 --vocab-size 320

# 3) Generate a product blurb using your trained checkpoint
python -m app.cli generate --name "Handmade Ceramic Mug" --category "Homeware" --features "matte glaze, 350 ml, comfortable handle" --tone "warm and elegant"
```

Default training is 1,200 steps. The `--steps` option is useful for short demos; more training and more examples usually improve output. Model checkpoint is `models/product_gpt.pt`.

## CLI reference
```text
python -m app.cli hello
python -m app.cli data
python -m app.cli train [--steps 1200] [--vocab-size 320]
python -m app.cli generate --name NAME [--category CATEGORY] [--features FEATURES] [--tone TONE] [--max-tokens 120] [--temperature 0.8] [--top-k 40
```

## Add your own training data
Each JSONL line in `data/products.jsonl` must contain `category`, `name`, and `description`. Use product text you wrote, have permission to use, or that is licensed for reuse. Keep examples focused on a single category for a coherent model. Example:
```json
{"category":"homeware","name":"Handmade Mug","description":"A glazed ceramic mug with a comfortable handle and a softly textured finish."}
```

## Tests
```powershell
pytest -q
```

## Implementation notes
- The tokenizer begins with all 256 byte values, then merges the most frequent adjacent pairs.
- The attention mask prevents each position from attending to future tokens.
- Training predicts the next token in the combined text stream.
- Temperature adjusts sampling randomness; top-k limits sampling to the most likely tokens.
- Checkpoints store model weights, configuration, and BPE merges. `.gitignore` keeps generated checkpoints and virtual environments out of Git.

## Limitations / responsible use
This tiny GPT is intentionally simple and has no factual grounding or guarantee of product-claim accuracy. Review generated copy before publishing, especially claims about materials, origin, durability, health, safety, or performance. Do not put private/customer information in the corpus.
