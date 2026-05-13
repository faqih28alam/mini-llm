# Mini-LLM (NanoGPT)

A minimal character-level Transformer built from scratch using PyTorch.

## Setup
1. Create a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate
2. Install dependencies:
   pip install -r requirements.txt

## Run
Run the training and generation script:
python nanogpt.py

This will:
1. Download the "Tiny Shakespeare" dataset.
2. Train a small Transformer model.
3. Generate sample text based on the trained model.
