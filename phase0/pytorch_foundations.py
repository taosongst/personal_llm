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

# Answer: t.view(8, 3) works fine — it gives you a (8, 3) tensor. The rule is
# that the total number of elements must stay the same. t has shape (2, 3, 4),
# so 2*3*4 = 24 elements. Any new shape whose dimensions multiply to 24 is
# valid: (8, 3), (6, 4), (24,), (1, 24), (2, 12), etc. If you tried
# t.view(8, 4) = 32 elements, that would crash with a RuntimeError because
# 32 ≠ 24. Think of view/reshape as just reinterpreting the same flat block of
# 24 numbers with different row/column boundaries.

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

# Answer: Gradients ACCUMULATE in .grad by default. Here's what happens step by step:
#   1. x = 3.0, x.grad = None
#   2. y = x² + 2x + 1 → dy/dx = 2x + 2 = 8
#   3. z = x³ + 1      → dz/dx = 3x² = 27
#   4. y.backward()     → x.grad = 8  (dy/dx)
#   5. z.backward()     → x.grad = 8 + 27 = 35  (accumulated! not replaced!)
#   6. print(x.grad)    → tensor(35.)
#
# This is why zero_grad() matters — without it, gradients from different
# backward() calls pile up. In training, each batch's gradients would add
# to the previous batch's, giving you nonsense updates.
#
# Note: z.backward() only works here because both y and z were built from x
# using separate computation graphs. If you tried y.backward() twice, it would
# crash because PyTorch frees the graph after the first backward() by default
# (unless you pass retain_graph=True).

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

# Answer: backward() only works on a SCALAR tensor (a tensor with a single number,
# shape []). If loss were a multi-element tensor like shape (4,), PyTorch wouldn't
# know what to differentiate — backward() computes d(scalar)/d(params).
#
# The convention: always reduce your loss to a single number before calling
# backward(). That's why we use:
#   - F.cross_entropy(...) which returns a scalar (it averages over all elements)
#   - Or .sum() / .mean() to reduce a tensor to a scalar
#
# In this example, loss = y.sum() collapses the (2,4) tensor into one number,
# so loss.backward() is valid. If you accidentally called y.backward() on the
# (2,4) tensor directly, you'd get:
#   RuntimeError: grad can be implicitly created only for scalar outputs

# --- Detaching from the graph ---
# Sometimes you want to use a tensor's value without tracking gradients.
z = y.detach()             # z has same values but no grad tracking
# Or use the context manager:
with torch.no_grad():
    z = x @ W              # no gradients computed — faster, less memory
    # Used during inference / evaluation

# Question: once detached, what happens to previous tracking? Is there any situation where we detach and need to re-attach?

# Answer: detach() creates a NEW tensor that shares the same data but is
# disconnected from the computation graph. The ORIGINAL tensor (y) is unchanged
# — it still has its grad_fn and is still part of the graph.
#
# Think of it like this: the graph is a chain of operations leading to y.
# z = y.detach() gives you a snapshot of y's values with the chain cut off.
# y itself still has the chain attached.
#
# Re-attaching: there's no "re-attach" operation. Once detached, z is just raw
# data. But you CAN feed z into new operations that DO track gradients:
#   z = y.detach()
#   w = z * 2  # w has no grad_fn — z was detached, so this chain starts cold
#   # But:
#   z.requires_grad_(True)  # manually turn tracking back on
#   w = z * 2               # now w.grad_fn exists, tracking from z onward
#
# Real use case: in GANs or reinforcement learning, you detach the output of
# one network so that gradients don't flow back through it when training
# another network. You're saying "treat this value as a fixed constant."


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

# Answer: LayerNorm needs the size because it has LEARNABLE parameters — gamma
# (scale) and beta (shift), each of shape (64,). One gamma and one beta per
# feature. After normalizing to mean=0, std=1, it does: output = gamma * normalized + beta.
# These let the network learn to undo the normalization if that's useful.
#
# If you used nn.LayerNorm(128) with a (2, 10, 64) input, it would crash:
#   RuntimeError: input shape [64] doesn't match normalized_shape [128]
# PyTorch checks that the last dimension(s) of the input match normalized_shape.
#
# You can also normalize over multiple trailing dims: nn.LayerNorm([10, 64])
# would normalize over the last TWO dimensions together. But in transformers,
# we always use LayerNorm(d_model) to normalize each token's feature vector
# independently.

# --- nn.Dropout(p) ---
# During training, randomly zeroes elements with probability p.
# Prevents overfitting. Disabled during eval (model.eval()).
drop = nn.Dropout(0.1)
x = torch.randn(2, 10, 64)
y = drop(x)                  # ~10% of values are 0, rest scaled up by 1/(1-p

# Question: explain why 1/(1-p) here, maybe explain from a 'keep some quantity unbiased' kind of way?

# Answer: Say you have a layer that outputs values summing to ~100 on average.
# With p=0.1, dropout zeroes out ~10% of them. Now the sum is only ~90.
# The next layer was trained expecting ~100, so it gets a weaker signal.
#
# Scaling by 1/(1-p) = 1/0.9 ≈ 1.11 compensates: the surviving 90% of values
# each get multiplied by 1.11, so 90 * 1.11 ≈ 100. The expected value is
# preserved — this is called "inverted dropout."
#
# Formally: for each element x_i, dropout outputs:
#   x_i * mask_i / (1-p)   where mask_i is 1 with prob (1-p), 0 with prob p
#   E[output_i] = x_i * (1-p) / (1-p) = x_i  ← unbiased!
#
# The big benefit: at eval time (model.eval()), dropout is completely turned off
# and you don't need any correction factor. The model just runs normally.
# Without the 1/(1-p) scaling during training, you'd need to scale DOWN by
# (1-p) at eval time instead, which is messier.

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

        # Answer: No broadcasting here — it's a straightforward reshape + loss computation.
        #
        # Step by step:
        #   1. logits has shape (4, 16, 100) — for each of 4 batches, 16 positions,
        #      100 scores (one per vocab word).
        #   2. logits.view(-1, 100) reshapes to (64, 100) — flatten batch and seq_len
        #      into one dimension. Now it's 64 "predictions", each with 100 class scores.
        #   3. targets has shape (4, 16). targets.view(-1) reshapes to (64,) — 64 integers,
        #      each is the correct class index (0-99).
        #   4. F.cross_entropy takes (N, C) logits and (N,) targets:
        #      - For each of the 64 positions, it applies softmax to the 100 scores
        #      - Then computes -log(probability of the correct class)
        #      - Returns the MEAN of all 64 losses → a single scalar
        #
        # We flatten because cross_entropy only accepts 2D logits. It doesn't know
        # about "batch" or "sequence" — it just sees N independent predictions.

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

# Answer: When you compute loss = model(x) → ... → F.cross_entropy(...), PyTorch
# builds a graph of every operation that led to `loss`. This graph stores
# references to all intermediate tensors (activations, weight matrices, etc.)
# so that backward() can compute gradients through them.
#
# If you store the loss TENSOR (not the number) in a list:
#   all_losses.append(loss)      # BAD — keeps the tensor object alive
# Python's garbage collector sees that `loss` is still referenced, so it can't
# free the computation graph. Every intermediate tensor from that forward pass
# stays in memory. Do this for 1000 batches and you've leaked 1000 full graphs.
#
# loss.item() extracts just the float (e.g., 2.34) — a plain Python number with
# no connection to the graph. The tensor can then be garbage collected, and the
# entire computation graph gets freed.
#   all_losses.append(loss.item())  # GOOD — just a number, graph is freed

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

# Answer: Your model outputs logits of shape (batch, seq_len, vocab_size),
# e.g. (4, 16, 50000). That's a 3D tensor. But F.cross_entropy expects:
#   - logits: (N, C) — N samples, C classes
#   - targets: (N,)  — N integers, each in [0, C)
#
# So you need to collapse the batch and seq_len dims into one:
#   logits.view(-1, vocab_size)  → (4*16, 50000) = (64, 50000)
#   targets.view(-1)             → (64,)
#
# Now each of the 64 positions is treated as an independent classification
# problem: "given these 50000 scores, the correct answer is targets[i]."
#
# Common bug: getting the order wrong. If you accidentally do
#   logits.view(-1, seq_len)  ← WRONG — C should be vocab_size, not seq_len
# the shapes might happen to work but the loss will be meaningless because
# you're treating sequence positions as classes.


# ============================================================================
# EXERCISE: Verify your understanding
# ============================================================================
# Before moving on, make sure you can answer these without looking above:
#
# 1. What's the difference between view() and reshape()? view() does not make copy, and requires contiguous memory.
#    Review: ✓ Correct. You could add: reshape() may copy if needed, and view() is preferred in LLM code because it's zero-cost.
#
# 2. What does requires_grad=True do?  track the gradient of this tensor
#    Review: ✓ Good enough. More precisely: it tells PyTorch to record all operations on this tensor
#    in a computation graph, so that when you call .backward(), it can compute d(loss)/d(this_tensor)
#    and store the result in .grad.
#
# 3. Why do we call optimizer.zero_grad()? clear previous computed gradient so we can train next batch
#    Review: ✓ Correct. Key detail: gradients accumulate by default (they ADD, not replace),
#    so without zeroing you'd be updating weights based on the sum of all past batches' gradients.
#
# 4. What does torch.tril() produce and why does GPT need it?  so that each token only attend itself or previous tokens
#    Review: ✓ Right idea. Be more specific: tril() produces a lower-triangular matrix of ones.
#    We use it to mask attention scores — the upper triangle gets filled with -inf, which becomes
#    0 after softmax, preventing tokens from "seeing the future."
#
# 5. What's the training loop order? (forward → ? → ? → ? → ?)  forward -> loss -> gradient -> update
#    Review: ✓ Close! The full sequence with function names is:
#    forward (logits = model(x)) → loss (F.cross_entropy) → backward (loss.backward()) →
#    clip (clip_grad_norm_) → step (optimizer.step()) → zero_grad (optimizer.zero_grad())
#    You're missing the clip and zero_grad steps, but the core idea is right.
#
# 6. What does nn.Embedding actually do under the hood? use the key (a tensor) to hash the embedding tensor
#    Review: ✗ Not quite — there's no hashing. It's a simple TABLE LOOKUP (array indexing).
#    nn.Embedding stores a weight matrix of shape (vocab_size, d_model). When you pass in
#    token_ids like [5, 10, 23], it literally does: weight[token_ids], which returns rows
#    5, 10, and 23. It's just fancy indexing — same as a Python list lookup, no hash function.
#
# 7. Why do we use .item() when logging the loss? otherwise the computation graph will be alive, and will be memory leakage
#    Review: ✓ Correct! Storing the tensor keeps a reference to the entire computation graph
#    in memory. .item() extracts a plain Python float, allowing the graph to be garbage collected.
#
# 8. What happens if your model is on GPU but your input tensor is on CPU? it will crash
#    Review: ✓ Correct. Specifically you get:
#    RuntimeError: Expected all tensors to be on the same device
#    Fix: input_ids = input_ids.to(device) before passing to model.
#
# When you're ready, tell me and I'll quiz you.
