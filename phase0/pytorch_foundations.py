"""
PyTorch Foundations for Building an LLM from Scratch
=====================================================

Read this file top to bottom. Run each section in a Python REPL or notebook
to see the outputs. Everything here is something you WILL use when building
your GPT.

Prerequisites assumed: Python, linear algebra, basic calculus.
"""

# ============================================================================
# 1. TENSORS — The fundamental data structure
# ============================================================================
# A tensor is just an n-dimensional array, like a numpy array but with:
#   - GPU support
#   - Automatic differentiation (autograd)

import torch

# --- Creating tensors ---
a = torch.tensor([1.0, 2.0, 3.0])              # from a list
b = torch.zeros(3, 4)                           # 3x4 of zeros
c = torch.ones(2, 3)                            # 2x3 of ones
d = torch.randn(2, 3)                           # 2x3 from standard normal — you'll use this A LOT
e = torch.arange(0, 10)                         # [0, 1, 2, ..., 9]
f = torch.linspace(0, 1, steps=5)               # [0.0, 0.25, 0.5, 0.75, 1.0]
g = torch.empty(2, 3)                           # uninitialized (fast, but garbage values)

# --- dtype matters ---
# LLM training uses float32 by default, float16/bfloat16 for mixed precision.
x = torch.randn(3, dtype=torch.float32)         # default
y = torch.randn(3, dtype=torch.bfloat16)        # half precision — saves memory, faster on modern GPUs
z = x.to(torch.bfloat16)                        # cast between types

# --- Shape, reshape, view ---
t = torch.randn(2, 3, 4)
print(t.shape)           # torch.Size([2, 3, 4])
print(t.ndim)            # 3
print(t.numel())         # 24 (total number of elements)

t2 = t.view(6, 4)        # reshape — must be contiguous in memory

# Question:what happens if t5 = t.view(8,3)? What's the rule here 


t3 = t.reshape(6, 4)     # reshape — works even if not contiguous (may copy)
t4 = t.view(2, -1)       # -1 means "infer this dimension" → shape (2, 12)

# --- Key difference: view vs reshape ---
# view() shares memory with the original (no copy). reshape() may or may not copy.
# In LLM code you'll see view() everywhere because it's zero-cost.


# ============================================================================
# 2. INDEXING AND SLICING
# ============================================================================
# Works like numpy. You need this for attention masks, selecting tokens, etc.

t = torch.randn(4, 5)
t[0]                      # first row
t[:, 0]                   # first column
t[1:3, 2:4]               # rows 1-2, cols 2-3
t[t > 0]                  # boolean indexing — all positive elements (flat)

# Fancy indexing
indices = torch.tensor([0, 2, 3])
t[indices]                # select rows 0, 2, 3

# This pattern appears in embedding lookups:
# embedding_matrix[token_ids]  →  looks up a row for each token


# ============================================================================
# 3. BROADCASTING
# ============================================================================
# When operating on tensors of different shapes, PyTorch "broadcasts" the
# smaller one to match. Rules:
#   - Dimensions are compared right-to-left
#   - A dimension of size 1 can be broadcast to any size
#   - Missing dimensions are treated as size 1

a = torch.randn(3, 4)    # (3, 4)
b = torch.randn(4)       # (4,) → treated as (1, 4) → broadcast to (3, 4)
c = a + b                 # works! each row of a gets b added to it

# This is how bias addition works in linear layers:
# output = input @ weight.T + bias
#   input:  (batch, in_features)
#   weight: (out_features, in_features)
#   bias:   (out_features,) → broadcasts across batch dimension


# ============================================================================
# 4. KEY OPERATIONS YOU'LL USE CONSTANTLY
# ============================================================================

# --- Matrix multiplication ---
# This is THE core operation. Attention = Q @ K.T @ V
a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = a @ b                 # (2, 3) @ (3, 4) → (2, 4)
c = torch.matmul(a, b)    # same thing
# @ also works for batched matmul:
# (B, M, K) @ (B, K, N) → (B, M, N)

# --- Transpose ---
a = torch.randn(2, 3)
a.T                        # simple transpose for 2D
a.transpose(0, 1)          # explicit — swap dims 0 and 1

# For higher dimensions (you'll need this for multi-head attention):
q = torch.randn(2, 8, 10, 64)   # (batch, heads, seq_len, head_dim)
k = q.transpose(-2, -1)          # swap last two dims → (2, 8, 64, 10)

# --- Softmax ---
# Converts raw scores to probabilities. Used in attention and final output.
logits = torch.randn(3, 5)
probs = torch.softmax(logits, dim=-1)   # along last dim; each row sums to 1
print(probs.sum(dim=-1))                # tensor([1., 1., 1.])

# --- Layer norm components ---
x = torch.randn(2, 5)
mean = x.mean(dim=-1, keepdim=True)     # keepdim keeps the shape compatible
std = x.std(dim=-1, keepdim=True)
normalized = (x - mean) / (std + 1e-5)  # this is what LayerNorm does internally

# --- Concatenation and stacking ---
a = torch.randn(2, 3)
b = torch.randn(2, 3)
torch.cat([a, b], dim=0)   # (4, 3) — concatenate along rows
torch.cat([a, b], dim=1)   # (2, 6) — concatenate along columns
torch.stack([a, b], dim=0)  # (2, 2, 3) — creates a NEW dimension

# --- Unsqueeze / squeeze ---
# Add or remove dimensions of size 1. Needed constantly for broadcasting.
a = torch.randn(3, 4)
a.unsqueeze(0).shape        # (1, 3, 4) — add batch dim
a.unsqueeze(-1).shape       # (3, 4, 1) — add trailing dim
b = torch.randn(1, 3, 1)
b.squeeze().shape            # (3,) — remove ALL size-1 dims
b.squeeze(0).shape           # (3, 1) — remove only dim 0


# ============================================================================
# 5. AUTOGRAD — Automatic differentiation
# ============================================================================
# This is how PyTorch computes gradients. You define a forward pass (compute
# the loss), call loss.backward(), and PyTorch fills in .grad for every
# parameter. You never write backprop by hand.


# Question
# Explain more about this gradient tracking, what happens if 
# x = torch.tensor(3.0, requires_grad=True)
# y = x ** 2 + 2 * x + 1  
# z = x**3 + 1
# y.backward()  
# z.backward()
# print(x.grad)    
# end of question

x = torch.tensor(3.0, requires_grad=True)  # "track this variable"
y = x ** 2 + 2 * x + 1                     # y = x² + 2x + 1
y.backward()                                # compute dy/dx
print(x.grad)                               # tensor(8.) — because dy/dx = 2x + 2 = 8

# --- With tensors ---
W = torch.randn(3, 4, requires_grad=True)
x = torch.randn(2, 3)
y = x @ W                                   # (2, 4)
loss = y.sum()                               # scalar
loss.backward()
print(W.grad.shape)                          # (3, 4) — same shape as W

# Question
# Does it make sense if loss is a tensor, and we do loss.backward()? If so, what's the convention here? 
# end of question 

# --- Detaching from the graph ---
# Sometimes you want to use a tensor's value without tracking gradients.
z = y.detach()             # z has same values but no grad tracking
# Or use the context manager:
with torch.no_grad():
    z = x @ W              # no gradients computed — faster, less memory
    # Used during inference / evaluation

# Question: once detached, what happens to previous tracking? Is there any situation where we detach and need to re-attach? 


# ============================================================================
# 6. nn.Module — How you build models
# ============================================================================
# Every model in PyTorch is a class that inherits from nn.Module.
# You define layers in __init__ and the forward pass in forward().

import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) — token IDs
        h = self.embedding(x)       # (batch, seq_len, d_model)
        logits = self.linear(h)     # (batch, seq_len, vocab_size)
        return logits

model = TinyModel(vocab_size=1000, d_model=64)

# --- Inspecting parameters ---
for name, param in model.named_parameters():
    print(f"{name:30s} {param.shape}")
# embedding.weight               torch.Size([1000, 64])
# linear.weight                  torch.Size([1000, 64])
# linear.bias                    torch.Size([1000])

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")


# ============================================================================
# 7. KEY nn LAYERS YOU'LL USE IN YOUR GPT
# ============================================================================

# --- nn.Embedding(num_embeddings, embedding_dim) ---
# A lookup table. Given token IDs, returns learned vectors.
emb = nn.Embedding(1000, 64)
token_ids = torch.tensor([5, 10, 23])
vectors = emb(token_ids)    # (3, 64) — one 64-dim vector per token
# Under the hood, this is just: emb.weight[token_ids]

# --- nn.Linear(in_features, out_features) ---
# y = x @ W.T + b
lin = nn.Linear(64, 128)
x = torch.randn(2, 10, 64)  # (batch, seq, features)
y = lin(x)                    # (2, 10, 128) — applies to last dimension

# --- nn.LayerNorm(normalized_shape) ---
# Normalizes across the last dimension(s) to have mean=0, std=1, then
# applies learnable scale (gamma) and shift (beta).
ln = nn.LayerNorm(64)
x = torch.randn(2, 10, 64)
y = ln(x)                    # (2, 10, 64) — each 64-dim vector is normalized

#Question: why do we need to specify a 64 in nn.LayerNorm(64) if the goal is to normalize the last dimension of x = torch.randn(2, 10, 64)? Can we do nn.LayerNorm(128) instead? 

# --- nn.Dropout(p) ---
# During training, randomly zeroes elements with probability p.
# Prevents overfitting. Disabled during eval (model.eval()).
drop = nn.Dropout(0.1)
x = torch.randn(2, 10, 64)
y = drop(x)                  # ~10% of values are 0, rest scaled up by 1/(1-p

# Question: explain why 1/(1-p) here, maybe explain from a 'keep some quantity unbiased' kind of way?

# --- nn.GELU() ---
# Activation function used in GPT (smoother than ReLU).
gelu = nn.GELU()
x = torch.randn(5)
y = gelu(x)                  # smooth nonlinearity, approximately x * sigmoid(1.702 * x)

# --- nn.ModuleList ---
# A list of modules that PyTorch can see (so their parameters are tracked).
layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(6)])
# Use this for your transformer blocks:
# self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])


# ============================================================================
# 8. THE TRAINING LOOP — Pattern you'll use every time
# ============================================================================
import torch.nn.functional as F

# Setup
model = TinyModel(vocab_size=100, d_model=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Fake data for illustration
input_ids = torch.randint(0, 100, (4, 16))     # (batch=4, seq_len=16)
# For language modeling, targets are inputs shifted by 1:
targets = torch.randint(0, 100, (4, 16))

# One training step
model.train()                                   # enable dropout etc.
logits = model(input_ids)                        # (4, 16, 100)
# Cross-entropy wants (N, C) and (N,), so reshape:
loss = F.cross_entropy(
    logits.view(-1, 100),                        # (4*16, 100)
    targets.view(-1)                             # (4*16,)
)
loss.backward()                                  # compute gradients
optimizer.step()                                 # update parameters
optimizer.zero_grad()                            # reset gradients for next step

# --- The full loop ---
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_idx in range(100):  # imagine 100 batches
        input_ids = torch.randint(0, 100, (4, 16))
        targets = torch.randint(0, 100, (4, 16))

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, 100), targets.view(-1))

        #Question
        # explain how does loss = F.cross_entropy(logits.view(-1, 100), targets.view(-1)) work exactly, is there any broadcasting here? 

        loss.backward()
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()       # .item() extracts the scalar value

    avg_loss = total_loss / 100
    print(f"Epoch {epoch}: loss = {avg_loss:.4f}")


# ============================================================================
# 9. GPU USAGE
# ============================================================================
# Move tensors and models to GPU with .to() or .cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TinyModel(vocab_size=100, d_model=32).to(device)
input_ids = torch.randint(0, 100, (4, 16)).to(device)

# EVERYTHING must be on the same device — model, inputs, targets.
# This is the #1 runtime error beginners hit:
#   RuntimeError: Expected all tensors to be on the same device

# Create tensors directly on GPU:
x = torch.randn(3, 4, device=device)


# ============================================================================
# 10. SAVING AND LOADING
# ============================================================================
# Save model weights (state dict), not the whole model object.

# Save
torch.save(model.state_dict(), "model.pt")

# Load
model2 = TinyModel(vocab_size=100, d_model=32)
model2.load_state_dict(torch.load("model.pt", weights_only=True))


# ============================================================================
# 11. MASKING — Critical for GPT's causal attention
# ============================================================================
# In GPT, each token can only attend to itself and previous tokens.
# This is enforced with a causal mask.

seq_len = 5

# Create a lower-triangular mask (True = allowed to attend)
mask = torch.tril(torch.ones(seq_len, seq_len))
print(mask)
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])

# In attention, you apply it like this:
attn_scores = torch.randn(seq_len, seq_len)     # raw Q @ K.T scores
attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
attn_weights = torch.softmax(attn_scores, dim=-1)
# The -inf positions become 0 after softmax → future tokens are ignored.

print(attn_weights)
# Each row sums to 1, and no token attends to future positions.


# ============================================================================
# 12. FUNCTIONAL vs MODULE API
# ============================================================================
# PyTorch offers two ways to do the same thing:

# Module (has learnable parameters, use in __init__):
ln = nn.LayerNorm(64)
y = ln(x)

# Functional (stateless, use in forward()):
y = F.softmax(logits, dim=-1)
y = F.cross_entropy(logits, targets)
y = F.gelu(x)
y = F.dropout(x, p=0.1, training=True)   # must pass training flag manually

# Rule of thumb:
#   - Has learnable parameters (weights)? → Use nn.Module version in __init__
#   - Pure computation (no parameters)?   → Use F.xxx in forward()


# ============================================================================
# 13. EINSUM — Optional but elegant
# ============================================================================
# Einstein summation notation. Concise way to express tensor operations.
# You'll see it in some attention implementations.

# Batch matrix multiply:  (B, M, K) @ (B, K, N) → (B, M, N)
a = torch.randn(2, 3, 4)
b = torch.randn(2, 4, 5)
c = torch.einsum('bmk,bkn->bmn', a, b)   # same as a @ b

# Dot product of each pair:
a = torch.randn(3, 4)
b = torch.randn(3, 4)
dots = torch.einsum('ij,ij->i', a, b)    # (3,) — row-wise dot products

# You don't NEED einsum — @ and .transpose() work fine — but it's good to
# recognize when you see it.


# ============================================================================
# 14. COMMON GOTCHAS
# ============================================================================

# 1. Forgetting optimizer.zero_grad()
#    Gradients ACCUMULATE by default. If you don't zero them, each step
#    uses the sum of all previous gradients. Always call zero_grad().

# 2. In-place operations on tensors that require grad
#    x += 1  can break autograd. Use x = x + 1 instead.

# 3. .item() for logging
#    loss.item() gives you a plain Python float. Don't store the tensor
#    itself — it keeps the computation graph alive and leaks memory.

# Question: what does it mean keeps the computation graph alive exactly

# 4. model.train() vs model.eval()
#    Dropout and batch norm behave differently during training vs eval.
#    Always call model.eval() before inference, model.train() before training.

# 5. Tensor on wrong device
#    If model is on GPU and input is on CPU → crash.
#    Always check: model.to(device), input.to(device), target.to(device).

# 6. Shape mismatches in cross_entropy
#    F.cross_entropy expects (N, C) logits and (N,) targets.
#    For language modeling: logits.view(-1, vocab_size), targets.view(-1)

# Question: explain this part a bit more 


# ============================================================================
# EXERCISE: Verify your understanding
# ============================================================================
# Before moving on, make sure you can answer these without looking above:
#
# 1. What's the difference between view() and reshape()? view() does not make copy, and requires contiguous memory. 
# 2. What does requires_grad=True do?  track the gradient of this tensor 
# 3. Why do we call optimizer.zero_grad()? clear previous computed gradient so we can train next batch 
# 4. What does torch.tril() produce and why does GPT need it?  so that each token only attend itself or previous tokens 
# 5. What's the training loop order? (forward → ? → ? → ? → ?)  forward -> loss -> gradient -> update 
# 6. What does nn.Embedding actually do under the hood? use the key (a tensor) to hash the embedding tensor
# 7. Why do we use .item() when logging the loss? otherwise the computation graph will be alive, and will be memory leakage 
# 8. What happens if your model is on GPU but your input tensor is on CPU? it will crash 
#
# When you're ready, tell me and I'll quiz you.
