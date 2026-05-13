# Project Context: Mini-LLM NanoGPT

## Overview
This project is a minimal implementation of a Large Language Model based on the Transformer architecture. It is designed for educational purposes to demonstrate how models like Claude and GPT work at a fundamental level.

## Core Technology
- Architecture: Character-level Transformer.
- Framework: PyTorch.
- Components: 
  - Self-Attention and Multi-Head Attention.
  - Position Embeddings.
  - Feed-Forward Networks.
  - Residual Connections and Layer Normalization.

## Data
- Dataset: Tiny Shakespeare input.txt.
- Tokenization: Character-level where each unique character maps to an integer.

## Files
- nanogpt.py: Contains data loading, model architecture, training loop, and text generation logic.
- requirements.txt: Python dependencies such as torch, requests, and numpy.
- README.md: Basic setup and run instructions.

## Key Concepts for the CLI
- This model predicts the next character given a sequence of previous characters.
- Hyperparameters are tuned for a nano scale to allow training on a CPU or standard laptop.
- The generate function uses the trained model to sample new text from a starting context.
