import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# --- Configuration & Hyperparameters (Must match training) ---
st.set_page_config(page_title="Mini-LLM: NanoGPT", page_icon="🧠")

BLOCK_SIZE = 64
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Model Architecture ---

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(DROPOUT))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa, self.ffwd = MultiHeadAttention(n_head, head_size), FeedFoward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        return logits, None
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- App Logic ---

@st.cache_resource
def load_resources():
    if not os.path.exists('input.txt'):
        return None, None, None, None
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    if not os.path.exists('model.pth'):
        return None, None, None, None
        
    model = BigramLanguageModel(vocab_size)
    model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model, stoi, itos, vocab_size

st.title("🧠 NanoGPT: Shakespeare Generator")
st.markdown("""
This is a character-level Transformer model trained on the works of Shakespeare.
Enter a prompt below and let the AI continue the dialogue!
""")

model, stoi, itos, vocab_size = load_resources()

if model is None:
    st.error("Missing `model.pth` or `input.txt`. Please train your model first using `python nanogpt.py`.")
else:
    with st.sidebar:
        st.header("Settings")
        tokens_to_gen = st.slider("Tokens to generate", 50, 1000, 300)
        temperature = st.slider("Creativity (Unused in this simple model)", 0.1, 1.0, 0.7) # Future proofing
        st.info(f"Running on: {DEVICE.upper()}")

    prompt = st.text_input("Start your Shakespearean scene:", value="ROMEO:")
    
    if st.button("Generate Text"):
        with st.spinner("The Bard is thinking..."):
            encode = lambda s: [stoi[c] for c in s if c in stoi]
            decode = lambda l: ''.join([itos[i] for i in l])
            
            # Handle empty prompt
            if not prompt:
                prompt = " "
                
            context = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
            
            # If prompt has characters not in vocab, context might be empty
            if context.shape[1] == 0:
                st.warning("Prompt contains characters not seen during training. Using a blank start instead.")
                context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

            generated_output = model.generate(context, max_new_tokens=tokens_to_gen)[0].tolist()
            result = decode(generated_output)
            
            st.subheader("Generated Result:")
            st.text_area(label="Generated text", value=result, height=400, label_visibility="collapsed")
            st.download_button("Download as .txt", result, file_name="shakespeare_gen.txt")
