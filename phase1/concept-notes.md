# Phase 1: Transformer Concepts — Q&A Notes

These are questions that came up while studying "Attention Is All You Need" and the transformer architecture.

---

## Encoder vs Decoder

**Q: What's the difference between encoder and decoder, and why does GPT use only the decoder?**

Both are stacks of transformer blocks, but they differ in **what each position can see**:

- **Encoder** — each position attends to **all** positions (bidirectional). Given "The cat sat on the mat", the word "sat" can see both "The cat" before it and "on the mat" after it. Rich representations, but can't be used for generation — it would be "cheating" by seeing the future.

- **Decoder** — each position can only attend to positions **before it** (causal/unidirectional). "sat" can see "The cat" but NOT "on the mat". Enforced by the causal mask (setting future positions to `-inf` before softmax).

In the original paper, they work together for **translation**:
1. Encoder reads the full input sentence (e.g., English) bidirectionally — fine because the input is fully known
2. Decoder generates the output (e.g., French) one token at a time, attending to its own previous outputs (masked self-attention) AND the encoder's representations (cross-attention)

**Why GPT uses decoder-only:**

1. **There's no separate "input" to encode.** In translation, you have a source sentence (encoder input) and a target sentence (decoder output). In language modeling, there's just one stream of text. The decoder reads and generates the same sequence.
2. **Simplicity.** One stack of layers instead of two. Fewer parameters for the same depth. No cross-attention sublayer needed.
3. **Generality.** A decoder-only model can do everything — it generates text, but it also "encodes" context through its causal attention over the prompt. Translation? Just prompt it with "Translate English to French: ..." — no architectural change needed.

**Tradeoff:** Decoder-only gives up bidirectional context. When processing token 5, it can't see tokens 6, 7, 8. BERT (encoder-only) exploits bidirectional context, making it better at classification and extraction where the full input is known. But for generation, decoder-only won.

---

## How Causal Masking Still "Understands" the Full Prompt

**Q: "When processing token 5, it can't see tokens 6, 7, 8 even though they're sitting right there in the prompt" — doesn't the model need to fully understand the prompt before generating?**

When you send a 7-token prompt, the model processes all 7 tokens in one forward pass, but with causal masking:

- Token 1 ("The") sees: `[The]`
- Token 2 ("cat") sees: `[The, cat]`
- Token 3 ("sat") sees: `[The, cat, sat]`
- ...
- Token 7 ("mat") sees: `[The, cat, sat, on, the, mat]` — **everything**

The model only uses the output at **position 7** (the last token) to generate the first new token. That position has attended to all previous tokens through multiple layers of attention, so by the final layer it has effectively integrated the entire prompt.

**Information flows forward through layers:**
```
Layer 1:  "sat" attends to "The, cat, sat"
Layer 2:  "on" attends to "The, cat, sat, on" — but "sat" already
          absorbed info from "The" and "cat" in Layer 1
...
Layer 32: The final position has very rich representations of the
          entire sequence built up through all previous layers
```

The real cost is mostly theoretical: when computing the representation of token 3 ("sat"), the model can't use tokens 4-7 to help understand what "sat" means. A bidirectional encoder would let "sat" attend to "on the mat" too. In practice, with enough layers and parameters, decoder-only models handle this just fine.

---

## Dimensionality Through the Transformer

**Q: What's the dimensionality of each layer? Use concrete numbers.**

Using: vocab=50,257 | seq_len (T)=7 | d_model=512 | n_heads=8 | d_k=64 | d_ff=2048 | N=6 layers

### Full shape flow:

```
Token IDs:             (7,)           ← just integers
After embedding:       (7, 512)       ← lookup in (50257, 512) table
After pos encoding:    (7, 512)       ← element-wise addition
  ┌──────────────────────────────────── × 6 blocks
  │ Q, K, V:           (7, 512) each  ← x @ W_Q/W_K/W_V, each (512, 512)
  │ Per-head Q,K,V:    (8, 7, 64)     ← reshape, 8 heads of 64 dims
  │ Attention scores:  (8, 7, 7)      ← Q_head @ K_head^T, this is the n² cost
  │ Attention weights: (8, 7, 7)      ← softmax(scores / sqrt(64))
  │ Head outputs:      (8, 7, 64)     ← weights @ V_head
  │ Concat + project:  (7, 512)       ← concat → (7, 512) @ W_O (512, 512)
  │ After residual+LN: (7, 512)
  │ FFN hidden:        (7, 2048)      ← x @ W_1 (512, 2048), momentary expansion
  │ FFN output:        (7, 512)       ← hidden @ W_2 (2048, 512), contract back
  │ After residual+LN: (7, 512)
  └──────────────────────────────────
Final projection:      (7, 50257)     ← @ W_vocab (512, 50257), one score per word
```

**Key insight:** 512 flows through the entire model unchanged. The only places dimensions differ are inside each head (64), inside FFN (2048), and at the output (50,257).

---

## Positional Encoding — From Intuition to Formula

**Q: Why the specific sinusoidal formula? It's confusing.**

### Building up from scratch:

**Naive approach** — add position number directly: `pos 0 → 0, pos 999 → 999`. Problems: wrong scale (overwhelms embeddings), no generalization, one dimension for position info.

**Better intuition — think about binary counting:**
```
Pos 0:  0 0 0 0
Pos 1:  0 0 0 1
Pos 2:  0 0 1 0
Pos 3:  0 0 1 1
Pos 4:  0 1 0 0
```

Each column alternates at a different **frequency** — rightmost bit flips every step (period 2), next every 2 steps (period 4), etc. Every position gets a unique pattern, nearby positions share most bits.

**Sinusoidal encoding = smooth, continuous version of binary.** Replace 0/1 bits with sin/cos waves at different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

| Dimension pair (i) | Divisor | Wavelength |
|---|---|---|
| i=0 | 1 | ~6 positions (fast, like lowest bit) |
| i=64 | 10 | ~63 positions |
| i=128 | 100 | ~628 positions |
| i=255 | 10,000 | ~62,832 positions (slow, like highest bit) |

Low dimensions change rapidly (fine position info), high dimensions change slowly (coarse position info). Every position gets a unique fingerprint.

### The relative position trick (why sin AND cos):

Using both sin and cos at each frequency means relative positions are a **fixed rotation**:

```
sin(pos+k) = sin(pos)·cos(k) + cos(pos)·sin(k)
cos(pos+k) = cos(pos)·cos(k) - sin(pos)·sin(k)

In matrix form:
[sin(pos+k)]   [cos(k)   sin(k)] [sin(pos)]
[cos(pos+k)] = [-sin(k)  cos(k)] [cos(pos)]
```

`PE(pos+k)` is a fixed rotation of `PE(pos)` that depends only on offset `k`, not on absolute position. This means:
```
PE(10) → PE(7):   rotation R(3)
PE(50) → PE(47):  same rotation R(3)
PE(999) → PE(996): same rotation R(3)
```

### How this helps attention learn relative position:

Attention scores are dot products `Q(pos_i) · K(pos_j)`. Since Q and K are derived from embeddings that include PE, the positional component of the dot product works out to:

```
sin(10)·sin(7) + cos(10)·cos(7) = cos(10-7) = cos(3)
sin(50)·sin(47) + cos(50)·cos(47) = cos(50-47) = cos(3)
```

The dot product **depends only on the offset**, not absolute positions. So `W_Q` and `W_K` only need to learn one set of weights to express "attend to the token 3 positions back" — it works at any position.

**Without this property** (e.g., random positional encodings), the relationship between position 10 and 7 would be completely different from 50 and 47. The model would need to memorize every absolute pair.

**Analogy:** It's like being on a ruler wanting to find the mark 3cm to your left. With sinusoidal encodings, "3cm left" is the same operation no matter where you stand. With random encodings, you'd need a different map for every starting position.

---

## RoPE and ALiBi — Modern Alternatives to Sinusoidal Encoding

**Q: What are RoPE and ALiBi?**

### RoPE (Rotary Position Embeddings)

Used by: LLaMA, Mistral, GPT-NeoX, most modern open-source LLMs.

Instead of **adding** positional information to embeddings at the start, apply **rotations directly to Q and K inside attention**:

```
# Sinusoidal: add PE to embeddings, then compute Q, K
x = embedding + PE(pos)
Q = x @ W_Q       ← position info mixed with token info early

# RoPE: compute Q, K first, then rotate by position
Q = x @ W_Q
K = x @ W_K
Q_rotated = rotate(Q, pos)
K_rotated = rotate(K, pos)
score = Q_rotated · K_rotated    ← position applied directly to attention
```

When you compute `Q_rotated(pos_i) · K_rotated(pos_j)`, the result depends only on `pos_i - pos_j` (relative distance) **by construction**, not just hopefully.

Why it's better:
- Relative positions **guaranteed** in the dot product, not just possible to learn
- Position and token information stay cleaner
- Empirically better, especially for longer sequences

### ALiBi (Attention with Linear Biases)

Used by: BLOOM, MPT.

No positional encoding at all. Instead, add a **linear penalty** to attention scores based on distance:

```
scores[i][j] = Q[i] · K[j] / sqrt(d_k)  -  m · |i - j|
```

Where `m` is a fixed slope per head (not learned). Nearby tokens preferred, distant tokens penalized. Different heads get different slopes — some attend far, some attend locally.

Why it's interesting:
- Zero extra parameters
- Extrapolates well to longer sequences (penalty just keeps growing linearly)
- Very simple to implement

### Comparison

| Method | Where applied | Relative position | Extrapolation | Used in |
|---|---|---|---|---|
| Sinusoidal | Added to embeddings | Possible (via rotation) | Theoretically yes | Original Transformer |
| Learned | Added to embeddings | Hopefully learned | No | GPT-2 |
| RoPE | Rotates Q, K in attention | Guaranteed | With extensions | LLaMA, Mistral |
| ALiBi | Bias on attention scores | Guaranteed | Naturally | BLOOM, MPT |

RoPE won as the dominant approach — cleanest way to make attention inherently relative-position-aware.
