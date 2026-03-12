# Phase 1: Transformer Architecture — Questions

Answer these from reasoning and what you've read/watched. After you've written your answers, we'll review them together before implementing.

---

## Embeddings

**Q1.** A token embedding maps each token ID to a vector. Why can't we just feed the raw token IDs (integers) directly into the network? What would be lost?

**Q2.** Transformers process all tokens in parallel (unlike RNNs). What problem does this create, and how do positional embeddings solve it? Why is position information necessary at all?

**Q3.** In the original "Attention Is All You Need" paper, they used sinusoidal positional encodings. GPT-2 uses learned positional embeddings instead. What is the difference? What is one advantage of each approach?

---

## Scaled Dot-Product Attention

**Q4.** Attention uses three vectors: Query (Q), Key (K), and Value (V). In plain English, what role does each play? Use an analogy if it helps.

**Q5.** The attention formula is `softmax(Q @ K.T / sqrt(d_k)) @ V`. Why do we divide by `sqrt(d_k)`? What goes wrong if we skip this scaling?

**Q6.** After computing `Q @ K.T`, we get a matrix of shape `(seq_len, seq_len)`. What does the value at position `(i, j)` represent *before* softmax? What about *after* softmax?

---

## Multi-Head Attention

**Q7.** Instead of doing one big attention computation with `d_model`-dimensional Q, K, V, multi-head attention splits into `n_heads` smaller attention computations. Why is this beneficial? What can multiple heads capture that a single head cannot?

**Q8.** If `d_model = 512` and `n_heads = 8`, what is `d_k` (the dimension per head)? How are Q, K, V split across heads — is it separate weight matrices per head, or one big matrix that gets reshaped?

**Q9.** After computing attention independently for each head, what do we do with the results? What is the purpose of the final linear projection (`W_O`) after concatenating the heads?

---

## Causal Masking

**Q10.** In a *language model* (as opposed to BERT), why must we prevent token `i` from seeing tokens at positions `j > i`? What would happen during training if we allowed the model to "see the future"?

**Q11.** We apply the causal mask *before* softmax, setting future positions to `-inf`. Why `-inf` specifically, and not `0` or some large negative number like `-1e9`? (Or does `-1e9` work too?)

---

## Feed-Forward Network (MLP Block)

**Q12.** Each transformer block has an attention sublayer followed by a feed-forward (MLP) sublayer. The MLP typically expands the dimension by 4x then contracts back: `d_model → 4*d_model → d_model`. Why expand and contract? Why not just use a single linear layer?

**Q13.** The original transformer uses ReLU in the MLP. GPT-2 uses GELU instead. What is GELU intuitively, and why might it work better than ReLU? (A rough description is fine — no need for the exact formula.)

---

## Layer Normalization

**Q14.** What does layer normalization do? Over which dimension does it normalize, and how does this differ from batch normalization?

**Q15.** GPT-2 uses "pre-norm" (LayerNorm *before* attention/MLP) while the original transformer uses "post-norm" (LayerNorm *after*). What practical difference does this make for training stability? Which is more common in modern LLMs?

---

## Residual Connections

**Q16.** What is a residual connection (skip connection), and what equation describes it? Why are residual connections critical for training deep networks — what problem do they solve?

**Q17.** In a transformer block, there are two residual connections: one around attention, one around the MLP. If the input to the block is `x`, write out the computation flow showing both residual connections and both layer norms (using pre-norm style).

---

## Putting It All Together

**Q18.** A GPT-style model stacks N transformer blocks. From input token IDs to output logits, list the full sequence of operations. (Embeddings → ... → logits)

**Q19.** During training on language modeling, the model predicts the *next* token at every position. If the input is `[The, cat, sat, on]`, what are the targets? How does the loss work across the full sequence?

**Q20.** At inference time (text generation), we have a prompt and want to generate new tokens one at a time. Describe the autoregressive generation loop. Why can't we generate all tokens in parallel like we process them during training?
