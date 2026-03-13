"""
Phase 2: Build a Tiny GPT from Scratch
=======================================

Fill in every TODO block. Each step has shape comments to guide you.
After completing each step, run the verification at the bottom of the file.

Steps:
  1. Data loading (char-level tokenizer + batching)
  2. CausalSelfAttention (multi-head, reshape approach)
  3. FeedForward + Block (FFN, pre-norm residuals)
  4. GPT model (full model: embeddings → blocks → logits)
  5. Training loop (overfit one batch, then full training)
  6. Text generation (greedy + temperature sampling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
batch_size = 32
block_size = 256       # context window (max sequence length)
n_layer = 4
n_head = 4
n_embd = 128           # embedding dimension (C)
dropout = 0.1
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# STEP 1: Data Loading
# =============================================================================
# Goal: Load Shakespeare, build a character-level tokenizer, create batches.

# --- Load the text ---
with open('phase2/data/input.txt', 'r') as f:
    text = f.read()

# TODO: Build the character-level tokenizer
# 1. Create `chars`: a sorted list of all unique characters in `text`
# 2. Set `vocab_size` to the number of unique characters
# 3. Create `stoi`: a dict mapping each character to its index (char -> int)
# 4. Create `itos`: a dict mapping each index to its character (int -> char)
# 5. Define `encode(s)`: takes a string, returns a list of integers
# 6. Define `decode(l)`: takes a list of integers, returns a string

# --- YOUR CODE HERE ---


# --- Prepare train/val splits ---
# TODO:
# 1. Encode the entire `text` into a tensor of integers called `data`
# 2. Split: first 90% → `train_data`, last 10% → `val_data`

# --- YOUR CODE HERE ---


def get_batch(split):
    """Sample a random batch of (input, target) pairs.

    Args:
        split: 'train' or 'val'

    Returns:
        x: (batch_size, block_size) tensor of input token indices
        y: (batch_size, block_size) tensor of target token indices (shifted by 1)
    """
    # TODO:
    # 1. Pick the right data source based on `split`
    # 2. Generate `batch_size` random starting indices (each in range [0, len(data) - block_size))
    # 3. For each starting index `i`, grab:
    #    - x: data[i : i + block_size]
    #    - y: data[i+1 : i + block_size + 1]
    # 4. Stack into tensors and move to `device`
    # Hint: use torch.randint and torch.stack

    # --- YOUR CODE HERE ---
    pass


# =============================================================================
# STEP 2: Multi-Head Causal Self-Attention
# =============================================================================
# Goal: Implement multi-head attention with the efficient reshape approach.
# One class does everything: Q/K/V projection, split into heads, attention, concat, output projection.

class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()
        assert n_embd % n_head == 0

        # TODO: Define these layers:
        # self.c_attn  = ...  # nn.Linear(n_embd, 3 * n_embd) — projects x into Q, K, V all at once
        # self.c_proj  = ...  # nn.Linear(n_embd, n_embd) — output projection (W_O)
        # self.attn_dropout  = ...  # nn.Dropout(dropout)
        # self.resid_dropout = ...  # nn.Dropout(dropout)

        # Register the causal mask as a buffer (not a parameter — no gradients)
        # This is a lower-triangular matrix of ones: shape (block_size, block_size)
        # self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size))
        #                                .view(1, 1, block_size, block_size))

        # --- YOUR CODE HERE ---
        pass

    def forward(self, x):
        """
        Args:
            x: (B, T, C) where C = n_embd

        Returns:
            out: (B, T, C)

        Shape flow:
            x: (B, T, C)
            → c_attn(x): (B, T, 3C)
            → split into q, k, v: each (B, T, C)
            → reshape each to (B, T, n_head, head_size) where head_size = C // n_head
            → transpose to (B, n_head, T, head_size)
            → att = q @ k.transpose(-2, -1): (B, n_head, T, T)
            → scale by (1.0 / sqrt(head_size))
            → apply causal mask: set upper-right triangle to -inf
            → softmax over last dim: (B, n_head, T, T)
            → dropout on attention weights
            → att @ v: (B, n_head, T, head_size)
            → transpose back to (B, T, n_head, head_size)
            → reshape (contiguous) to (B, T, C)
            → c_proj: (B, T, C)
            → residual dropout
        """
        B, T, C = x.size()
        head_size = C // n_head

        # TODO: Implement the full attention forward pass following the shape flow above.
        # Key operations:
        #   self.c_attn(x)              → (B, T, 3C), then .split(C, dim=2) → three (B, T, C) tensors
        #   .view(B, T, n_head, head_size).transpose(1, 2) → (B, n_head, T, head_size)
        #   q @ k.transpose(-2, -1)     → (B, n_head, T, T)
        #   .masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #   F.softmax(att, dim=-1)
        #   att @ v                     → (B, n_head, T, head_size)
        #   .transpose(1, 2).contiguous().view(B, T, C)  → (B, T, C)

        # --- YOUR CODE HERE ---
        pass


# =============================================================================
# STEP 3: FeedForward + Block
# =============================================================================

class FeedForward(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU, contract, dropout."""

    def __init__(self):
        super().__init__()
        # TODO: Define the layers:
        # A sequential network: Linear(n_embd → 4*n_embd) → GELU → Linear(4*n_embd → n_embd) → Dropout
        # Hint: nn.Sequential makes this clean

        # --- YOUR CODE HERE ---
        pass

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            out: (B, T, C)
        """
        # TODO: pass x through the sequential network

        # --- YOUR CODE HERE ---
        pass


class Block(nn.Module):
    """Transformer block: pre-norm attention + pre-norm FFN, both with residual connections.

    Computation flow (pre-norm style):
        x' = x + CausalSelfAttention(LayerNorm(x))
        x'' = x' + FeedForward(LayerNorm(x'))
    """

    def __init__(self):
        super().__init__()
        # TODO: Define these layers:
        # self.ln1  = ...  # LayerNorm(n_embd)
        # self.attn = ...  # CausalSelfAttention()
        # self.ln2  = ...  # LayerNorm(n_embd)
        # self.ffn  = ...  # FeedForward()

        # --- YOUR CODE HERE ---
        pass

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            out: (B, T, C)
        """
        # TODO: Implement the two-line pre-norm residual pattern:
        #   x = x + self.attn(self.ln1(x))
        #   x = x + self.ffn(self.ln2(x))

        # --- YOUR CODE HERE ---
        pass


# =============================================================================
# STEP 4: Full GPT Model
# =============================================================================

class GPT(nn.Module):
    """
    Full GPT model.

    Architecture:
        token_emb(idx)              → (B, T, C)
        + pos_emb(positions)        → (B, T, C)
        dropout                     → (B, T, C)
        Block × n_layer             → (B, T, C)
        LayerNorm                   → (B, T, C)
        Linear(C → vocab_size)      → (B, T, vocab_size)
    """

    def __init__(self):
        super().__init__()
        # TODO: Define these layers:
        # self.token_emb = ...  # nn.Embedding(vocab_size, n_embd)
        # self.pos_emb   = ...  # nn.Embedding(block_size, n_embd) — learned positional embeddings
        # self.drop      = ...  # nn.Dropout(dropout)
        # self.blocks    = ...  # nn.Sequential(*[Block() for _ in range(n_layer)])
        # self.ln_f      = ...  # LayerNorm(n_embd) — final layer norm
        # self.lm_head   = ...  # nn.Linear(n_embd, vocab_size, bias=False)

        # --- YOUR CODE HERE ---
        pass

    def forward(self, idx, targets=None):
        """
        Args:
            idx:     (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices, or None

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar loss if targets provided, else None

        Shape flow:
            idx: (B, T)
            → token_emb: (B, T, C)    — look up each token index
            → + pos_emb: (B, T, C)    — add positional embeddings for positions 0..T-1
            → dropout: (B, T, C)
            → blocks: (B, T, C)       — pass through all transformer blocks
            → ln_f: (B, T, C)         — final layer norm
            → lm_head: (B, T, vocab_size)  — project to vocabulary
        """
        B, T = idx.size()

        # TODO: Implement the forward pass following the shape flow above.
        # For positional embeddings: create a position tensor [0, 1, 2, ..., T-1]
        #   pos = torch.arange(0, T, device=device)  → shape (T,)
        #   self.pos_emb(pos) broadcasts across the batch
        #
        # For the loss (only when targets is not None):
        #   logits shape is (B, T, vocab_size), targets shape is (B, T)
        #   F.cross_entropy expects (N, C) and (N,) so reshape:
        #     loss = F.cross_entropy(logits.view(B*T, vocab_size), targets.view(B*T))

        # --- YOUR CODE HERE ---
        pass


# =============================================================================
# STEP 5: Training Loop
# =============================================================================

@torch.no_grad()
def estimate_loss(model):
    """Estimate train and val loss by averaging over `eval_iters` batches.

    Returns:
        dict with keys 'train' and 'val', each mapping to a float loss value.
    """
    # TODO:
    # 1. Set model to eval mode
    # 2. For each split ('train', 'val'):
    #    a. Run `eval_iters` forward passes with get_batch(split)
    #    b. Accumulate the losses
    #    c. Store the mean loss
    # 3. Set model back to train mode
    # 4. Return the dict

    # --- YOUR CODE HERE ---
    pass


def train():
    """Main training function."""
    model = GPT().to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # TODO:
    # 1. Create an AdamW optimizer with `learning_rate`
    #    Hint: torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #
    # 2. Training loop for `max_iters` steps:
    #    a. Every `eval_interval` steps, call estimate_loss() and print train/val loss
    #    b. Sample a batch with get_batch('train')
    #    c. Forward pass: logits, loss = model(x, y)
    #    d. Zero gradients: optimizer.zero_grad(set_to_none=True)
    #    e. Backward pass: loss.backward()
    #    f. Update weights: optimizer.step()
    #
    # 3. After training, print final losses

    # --- YOUR CODE HERE ---
    pass


# =============================================================================
# STEP 6: Text Generation
# =============================================================================
# Add this method to the GPT class (or define it here and call it on the model).

def generate(model, idx, max_new_tokens, temperature=1.0):
    """Generate new tokens autoregressively.

    Args:
        model:          the GPT model (in eval mode)
        idx:            (B, T) tensor of starting token indices (the prompt)
        max_new_tokens: how many new tokens to generate
        temperature:    > 1.0 = more random, < 1.0 = more deterministic, 0 = greedy

    Returns:
        idx: (B, T + max_new_tokens) tensor with generated tokens appended

    Algorithm:
        For each new token:
        1. Crop idx to the last `block_size` tokens (model can't handle longer)
        2. Forward pass → logits: (B, T, vocab_size)
        3. Take logits at the last position: (B, vocab_size)
        4. Apply temperature: divide logits by temperature
        5. Convert to probabilities: softmax
        6. Sample from the distribution: torch.multinomial (or argmax if temperature ≈ 0)
        7. Append the sampled token to idx
    """
    # TODO: Implement the autoregressive generation loop.
    # Hint: use model.eval() before generating, and wrap in torch.no_grad()

    # --- YOUR CODE HERE ---
    pass


# =============================================================================
# VERIFICATION & MAIN
# =============================================================================

if __name__ == '__main__':
    # --- Step 1 verification ---
    # Uncomment after completing Step 1:
    # xb, yb = get_batch('train')
    # print(f"x shape: {xb.shape}, y shape: {yb.shape}")  # expect (32, 256) each
    # print(f"First batch decoded: {decode(xb[0].tolist()[:50])}")
    # print(f"Vocab size: {vocab_size}")  # expect 65

    # --- Step 2 verification ---
    # Uncomment after completing Step 2:
    # attn = CausalSelfAttention().to(device)
    # test_x = torch.randn(2, 16, n_embd, device=device)
    # print(f"Attention output shape: {attn(test_x).shape}")  # expect (2, 16, 128)

    # --- Step 3 verification ---
    # Uncomment after completing Step 3:
    # block = Block().to(device)
    # test_x = torch.randn(2, 16, n_embd, device=device)
    # print(f"Block output shape: {block(test_x).shape}")  # expect (2, 16, 128)

    # --- Step 4 verification ---
    # Uncomment after completing Step 4:
    # model = GPT().to(device)
    # test_idx = torch.zeros(2, 16, dtype=torch.long, device=device)
    # logits, loss = model(test_idx, test_idx)
    # print(f"Logits shape: {logits.shape}")  # expect (2, 16, vocab_size)
    # print(f"Loss: {loss.item():.2f}")  # expect ~4.17 = -ln(1/65) for random init

    # --- Step 5: Train ---
    # Uncomment after completing Step 5:
    # train()

    # --- Step 6: Generate ---
    # Uncomment after completing Step 6:
    # model = GPT().to(device)
    # # Load trained weights if you saved them, or run after train()
    # prompt = "To be or not to be"
    # idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    # generated = generate(model, idx, max_new_tokens=200, temperature=0.8)
    # print(decode(generated[0].tolist()))

    print("Skeleton loaded. Start filling in Step 1!")
