# Phase 1: Transformer Architecture — Questions

Answer these from reasoning and what you've read/watched. After you've written your answers, we'll review them together before implementing.

---

## Embeddings

**Q1.** A token embedding maps each token ID to a vector. Why can't we just feed the raw token IDs (integers) directly into the network? What would be lost?

If we use token id directly then the overall representation is 1 dimensional and we loss a lot of information about how different tokens are related, and we might accidentally impose a lot of unexpected relations. For example, if apple=1, banana=2, king=3,queue=4,toilet=5, then we are saying banana-apple=queen-king=toilet-queen, which doesn't make sense.

> **Review: Correct.** Great example of false ordinal relationships. Small clarification: more precisely, integers impose a single arbitrary ordering with no meaningful geometry. Embeddings give each token a rich vector where distance and direction encode semantic relationships.

**Q2.** Transformers process all tokens in parallel (unlike RNNs). What problem does this create, and how do positional embeddings solve it? Why is position information necessary at all?
One problem is that there is no such concept as 'position' or 'relative position'. Position embedding adds position information into the representation, and allows the model to learn a fixed linear representation between PE(pos) and PE(pos+k) for each fixed k. It matters as this is natually the case for our language.

> **Review: Correct but imprecise.** You jumped to the rotation/linear transformation property, which is specific to sinusoidal encodings, not positional embeddings in general. The core answer is simpler: without position info, "The cat sat" and "sat cat The" produce identical outputs because attention is a set operation (order-invariant). Positional embeddings break this symmetry.

**Q3.** In the original "Attention Is All You Need" paper, they used sinusoidal positional encodings. GPT-2 uses learned positional embeddings instead. What is the difference? What is one advantage of each approach?
Sinusoidal positional encodings allows the model to learn the actual linear transformation between PE(pos) and PE(pos+k), no extra parameters needed and it generalize when the max_len extends; learned positional embeddings is more flexible but does not generalize when max_len extends.

> **Review: Correct.** Good contrast on generalization vs flexibility. Could also mention sinusoidal requires zero learned parameters.

---

## Scaled Dot-Product Attention

**Q4.** Attention uses three vectors: Query (Q), Key (K), and Value (V). In plain English, what role does each play? Use an analogy if it helps.
Key: which word I am searching for
Query: which dictionary I am using
Value: after searching, what do I do with it.

> **Review: Q and K are swapped.** Query = "What am I looking for?" (the token doing the searching). Key = "What do I contain?" (the token being searched against). Value = "What information do I provide if matched?" Think search engine: you type a **query**, it matches against **keys** (page titles), and returns **values** (page content). The query comes from the current token, keys come from all tokens being attended to.

**Q5.** The attention formula is `softmax(Q @ K.T / sqrt(d_k)) @ V`. Why do we divide by `sqrt(d_k)`? What goes wrong if we skip this scaling?
To prevent the overall value from getting to big; if too big, the values lie inside a position that the gradient of softmax w.r.t. logits become very small.

> **Review: Correct.** Good on the gradient saturation point. To strengthen: when `d_k` is large, the dot products have variance ~`d_k` (each element contributes ~1 to the variance), so dividing by `sqrt(d_k)` brings variance back to ~1.

**Q6.** After computing `Q @ K.T`, we get a matrix of shape `(seq_len, seq_len)`. What does the value at position `(i, j)` represent *before* softmax? What about *after* softmax?

represent how much the token at i should care about the token at j

I think of softmax just as turning logits into actual probability; it's a mathematical way of normalizing it.

> **Review: Right intuition, but the question asks you to distinguish before vs after.** Before softmax: raw compatibility score (unbounded, any real number). Higher = more compatible, but values aren't comparable across rows. After softmax: a probability weight (0 to 1, each row sums to 1). Now it's an actual proportion — "token i allocates 40% of its attention to token j."

---

## Multi-Head Attention

**Q7.** Instead of doing one big attention computation with `d_model`-dimensional Q, K, V, multi-head attention splits into `n_heads` smaller attention computations. Why is this beneficial? What can multiple heads capture that a single head cannot?

The computational cost are the same, while allow different head to capture different patterns to attend to.

another benefits is that for things like ALiBi, multihead allows different penalty parameters for different heads.

> **Review: Good but incomplete.** Equal cost point is correct. The ALiBi point is creative but tangential (came years later). The core answer should emphasize: a single head computes one attention pattern (one weighted average). Multiple heads let the model attend to different things simultaneously — one head for syntactic relationships, another for nearby tokens, another for semantic similarity, etc. A single head must compromise between all of these.

**Q8.** If `d_model = 512` and `n_heads = 8`, what is `d_k` (the dimension per head)? How are Q, K, V split across heads — is it separate weight matrices per head, or one big matrix that gets reshaped?

d_k = d_model/n_heads = 64

Actually I am not sure; each Q has dimension d_k \times d_k, and we have n_heads copies of it, does not sounds like reshaping a big matrix;

> **Review: d_k correct. Weight matrix part is wrong.** In practice, it's **one big matrix that gets reshaped**. You compute `Q = x @ W_Q` where `W_Q` is `(512, 512)`, then reshape the output from `(seq_len, 512)` to `(n_heads, seq_len, 64)`. This is mathematically equivalent to 8 separate `(512, 64)` matrices, but one big matmul is much faster on GPUs. Important implementation detail you'll need when coding it.

**Q9.** After computing attention independently for each head, what do we do with the results? What is the purpose of the final linear projection (`W_O`) after concatenating the heads?
We concatenate the results, and go through a feed forward layer to produce logits.
The final linear projection gives it more space to learn the pattern (i.e. making the function more complex).

> **Review: Wrong on both parts.** After concatenation, the result goes through `W_O` (a linear projection), **not** a feed-forward layer that produces logits. The FFN is a separate sublayer that comes after. `W_O` maps `(seq_len, 512)` → `(seq_len, 512)`. The purpose of `W_O` is to **mix information across heads**. Without it, heads are completely independent. `W_O` lets the model combine what different heads learned (e.g., "head 1 found the subject, head 3 found the verb" → `W_O` combines these into a unified representation).

---

## Causal Masking

**Q10.** In a *language model* (as opposed to BERT), why must we prevent token `i` from seeing tokens at positions `j > i`? What would happen during training if we allowed the model to "see the future"?

The causal masking. If we allow the model to see the futures that's target leakage, in other words we allow the model to cheat.

> **Review: Correct.** Target leakage is exactly right. To add: the model would learn to just copy the next token from the input instead of learning to predict, so it would get near-zero training loss but generate garbage at inference time (where future tokens don't exist).

**Q11.** We apply the causal mask *before* softmax, setting future positions to `-inf`. Why `-inf` specifically, and not `0` or some large negative number like `-1e9`? (Or does `-1e9` work too?)

because softmax turns -inf to 0
Actually I think in practice -1e9 might work as well? Not sure what kind of problems it will cause

> **Review: Correct.** `-1e9` works fine in practice and is what most implementations actually use. True `-inf` can cause NaN issues with some floating point operations. The key insight is that any sufficiently large negative number makes softmax output effectively 0.

---

## Feed-Forward Network (MLP Block)

**Q12.** Each transformer block has an attention sublayer followed by a feed-forward (MLP) sublayer. The MLP typically expands the dimension by 4x then contracts back: `d_model → 4*d_model → d_model`. Why expand and contract? Why not just use a single linear layer?

I think of it just as a way to give the network more complexity / capabilities to fit more complex functions.

> **Review: Too vague.** Two specific points you should know: (1) A single linear layer can only compute **linear transformations** — no matter how wide, `x @ W` is linear. You need the nonlinearity (ReLU/GELU) between two linear layers to approximate nonlinear functions. That's the fundamental reason for two layers. (2) The expansion to 4x gives the network a higher-dimensional space to compute in before compressing back — project up to a space where features can be independently activated/deactivated, then project back.

**Q13.** The original transformer uses ReLU in the MLP. GPT-2 uses GELU instead. What is GELU intuitively, and why might it work better than ReLU? (A rough description is fine — no need for the exact formula.)

It's a convolution of Relu with a smooth kernel (that concentrated around 0), overall just to make the function smoother.

> **Review: Not quite right.** GELU is `x · Φ(x)` where `Φ` is the standard normal CDF. Intuitively, it **softly gates** each element: values that are clearly positive pass through (~ReLU), values that are clearly negative get zeroed (~ReLU), but values near zero get **partially** passed through based on their magnitude. ReLU has a hard cutoff at 0 which can cause "dead neurons." GELU's smooth transition avoids this.

---

## Layer Normalization

**Q14.** What does layer normalization do? Over which dimension does it normalize, and how does this differ from batch normalization?

layer normalization normalize the hidden features of each input token; batch normalization normalizes each feature among all the tokens within the same batch.

> **Review: Correct.** Clear and concise.

**Q15.** GPT-2 uses "pre-norm" (LayerNorm *before* attention/MLP) while the original transformer uses "post-norm" (LayerNorm *after*). What practical difference does this make for training stability? Which is more common in modern LLMs?

Not sure, but my intuition is that normalize before attention/MLP is better as it makes distribution before feeding into attention more stable and easier to train.

> **Review: Right intuition, incomplete on the "why."** Pre-norm is better for stability because the residual path `x + Sublayer(LayerNorm(x))` means the gradient flows through the `+ x` path unimpeded — it's always the identity plus something. With post-norm `LayerNorm(x + Sublayer(x))`, the gradient must pass through LayerNorm, which can distort it. Pre-norm is standard in modern LLMs.

---

## Residual Connections

**Q16.** What is a residual connection (skip connection), and what equation describes it? Why are residual connections critical for training deep networks — what problem do they solve?

x = x+MLP(x) or x = x+attention(x)
It resolves the vanishing gradient issue

> **Review: Correct but too brief.** Explain *how* it solves vanishing gradients: during backprop, the gradient of `x + f(x)` with respect to `x` is `1 + f'(x)`. That `+ 1` term means the gradient is **at least 1** along the skip path, so it can flow through many layers without shrinking to zero.

**Q17.** In a transformer block, there are two residual connections: one around attention, one around the MLP. If the input to the block is `x`, write out the computation flow showing both residual connections and both layer norms (using pre-norm style).
x = x+MLP(x) or x = x+attention(x)

> **Review: Incomplete — this just restates Q16.** The question asks for the full flow with both sublayers. The answer should be:
> ```
> x' = x + Attention(LayerNorm(x))
> x'' = x' + FFN(LayerNorm(x'))
> ```
> This shows: normalize → sublayer → add back input, done twice sequentially. You'll be coding this exact flow, so make sure it's precise.

---

## Putting It All Together

**Q18.** A GPT-style model stacks N transformer blocks. From input token IDs to output logits, list the full sequence of operations. (Embeddings → ... → logits)
embeddings -> add positional encodings -> attention block (multihead attention +  residual + layer norm + feed forward + residual + layer norm) repeat 6 times -> final output projection = logits

> **Review: Mostly correct.** Minor note: in pre-norm style, there's typically a **final LayerNorm** after the last block and before the output projection. So: `... → final LayerNorm → output projection → logits`.

**Q19.** During training on language modeling, the model predicts the *next* token at every position. If the input is `[The, cat, sat, on]`, what are the targets? How does the loss work across the full sequence?

The targets are the actual next token within it's training sets. The loss is an average of log loss for all the tokens marked as targets.

> **Review: Correct but should be more specific.** The targets are the shifted input sequence. Specifically: input = `[The, cat, sat, on]`, targets = `[cat, sat, on, <next_token>]`. Each position predicts the token one step ahead. Loss = average cross-entropy across all 4 positions.

**Q20.** At inference time (text generation), we have a prompt and want to generate new tokens one at a time. Describe the autoregressive generation loop. Why can't we generate all tokens in parallel like we process them during training?
If we generate all the tokens paralelly it might not be consistent (i.e. may not read like an actual sentence) as we didn't impose dependencies between these tokens other than the fact that the prompt is the same prompt.

> **Review: Wrong reasoning.** The reason isn't about consistency or coherence — it's a **causal data dependency**: to generate token 8, you need token 7 as input. But token 7 doesn't exist until you generate it. And to generate token 7, you need token 6. So you **must** go one at a time. During training, all tokens already exist (they're from the training data), so you can process them in parallel with masking. During inference, future tokens literally don't exist yet.

---

## Summary

| Rating | Questions |
|---|---|
| Solid | Q1, Q2, Q3, Q5, Q10, Q11, Q14, Q18 |
| Right idea, needs depth | Q6, Q7, Q12, Q15, Q16, Q19 |
| Wrong or significantly off | Q4 (swapped Q/K), Q8 (weight matrix), Q9 (W_O purpose), Q13 (GELU), Q17 (incomplete), Q20 (wrong reasoning) |

**Key areas to revisit before implementing:**
1. Q/K/V roles (Q4) — get Q and K the right way around
2. Multi-head implementation (Q8) — one big matrix, reshaped, not separate matrices
3. W_O purpose (Q9) — mixes across heads, not a FFN producing logits
4. Full block flow (Q17) — `x' = x + Attn(LN(x))`, `x'' = x' + FFN(LN(x'))` — you'll code this
5. Autoregressive constraint (Q20) — it's a data dependency, not a quality issue
