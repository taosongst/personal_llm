# Phase 1: Transformer Architecture — Questions

Answer these from reasoning and what you've read/watched. After you've written your answers, we'll review them together before implementing.

---

## Embeddings

**Q1.** A token embedding maps each token ID to a vector. Why can't we just feed the raw token IDs (integers) directly into the network? What would be lost?

If we use token id directly then the overall representation is 1 dimensional and we loss a lot of information about how different tokens are related, and we might accidentally impose a lot of unexpected relations. For example, if apple=1, banana=2, king=3,queue=4,toilet=5, then we are saying banana-apple=queue-king=toilet-queue, which doesn't make sense. 

**Q2.** Transformers process all tokens in parallel (unlike RNNs). What problem does this create, and how do positional embeddings solve it? Why is position information necessary at all?
One problem is that there is no such concept as 'position' or 'relative position'. Position embedding adds position information into the representation, and allows the model to learn a fixed linear representation between PE(pos) and PE(pos+k) for each fixed k. It matters as this is natually the case for our language. 

**Q3.** In the original "Attention Is All You Need" paper, they used sinusoidal positional encodings. GPT-2 uses learned positional embeddings instead. What is the difference? What is one advantage of each approach?
Sinusoidal positional encodings allows the model to learn the actual linear transformation between PE(pos) and PE(pos+k), no extra parameters needed and it generalize when the max_len extends; learned positional embeddings is more flexible but does not generalize when max_len extends. 
---

## Scaled Dot-Product Attention

**Q4.** Attention uses three vectors: Query (Q), Key (K), and Value (V). In plain English, what role does each play? Use an analogy if it helps.
Key: which word I am searching for 
Query: which dictionary I am using 
Value: after searching, what do I do with it. 

**Q5.** The attention formula is `softmax(Q @ K.T / sqrt(d_k)) @ V`. Why do we divide by `sqrt(d_k)`? What goes wrong if we skip this scaling?
To prevent the overall value from getting to big; if too big, the values lie inside a position that the gradient of softmax w.r.t. logits become very small. 

**Q6.** After computing `Q @ K.T`, we get a matrix of shape `(seq_len, seq_len)`. What does the value at position `(i, j)` represent *before* softmax? What about *after* softmax?

represent how much the token at i should care about the token at j

I think of softmax just as turning logits into actual probability; it's a mathematical way of normalizing it. 
---

## Multi-Head Attention

**Q7.** Instead of doing one big attention computation with `d_model`-dimensional Q, K, V, multi-head attention splits into `n_heads` smaller attention computations. Why is this beneficial? What can multiple heads capture that a single head cannot?

The computational cost are the same, while allow different head to capture different patterns to attend to. 

another benefits is that for things like ALiBi, multihead allows different penalty parameters for different heads. 

**Q8.** If `d_model = 512` and `n_heads = 8`, what is `d_k` (the dimension per head)? How are Q, K, V split across heads — is it separate weight matrices per head, or one big matrix that gets reshaped?

d_k = d_model/n_heads = 64

Actually I am not sure; each Q has dimension d_k \times d_k, and we have n_heads copies of it, does not sounds like reshaping a big matrix;

**Q9.** After computing attention independently for each head, what do we do with the results? What is the purpose of the final linear projection (`W_O`) after concatenating the heads?
We concatenate the results, and go through a feed forward layer to produce logits.
The final linear projection gives it more space to learn the pattern (i.e. making the function more complex). 

---

## Causal Masking

**Q10.** In a *language model* (as opposed to BERT), why must we prevent token `i` from seeing tokens at positions `j > i`? What would happen during training if we allowed the model to "see the future"?

The causal masking. If we allow the model to see the futures that's target leakage, in other words we allow the model to cheat.

**Q11.** We apply the causal mask *before* softmax, setting future positions to `-inf`. Why `-inf` specifically, and not `0` or some large negative number like `-1e9`? (Or does `-1e9` work too?)

because softmax turns -inf to 0
Actually I think in practice -1e9 might work as well? Not sure what kind of problems it will cause
---

## Feed-Forward Network (MLP Block)

**Q12.** Each transformer block has an attention sublayer followed by a feed-forward (MLP) sublayer. The MLP typically expands the dimension by 4x then contracts back: `d_model → 4*d_model → d_model`. Why expand and contract? Why not just use a single linear layer?

I think of it just as a way to give the network more complexity / capabilities to fit more complex functions. 

**Q13.** The original transformer uses ReLU in the MLP. GPT-2 uses GELU instead. What is GELU intuitively, and why might it work better than ReLU? (A rough description is fine — no need for the exact formula.)

It's a convolution of Relu with a smooth kernel (that concentrated around 0), overall just to make the function smoother. 
---

## Layer Normalization

**Q14.** What does layer normalization do? Over which dimension does it normalize, and how does this differ from batch normalization?

layer normalization normalize the hidden features of each input token; batch normalization normalizes each feature among all the tokens within the same batch. 

**Q15.** GPT-2 uses "pre-norm" (LayerNorm *before* attention/MLP) while the original transformer uses "post-norm" (LayerNorm *after*). What practical difference does this make for training stability? Which is more common in modern LLMs?

Not sure, but my intuition is that normalize before attention/MLP is better as it makes distribution before feeding into attention more stable and easier to train. 
---

## Residual Connections

**Q16.** What is a residual connection (skip connection), and what equation describes it? Why are residual connections critical for training deep networks — what problem do they solve?

x = x+MLP(x) or x = x+attention(x)
It resolves the vanishing gradient issue

**Q17.** In a transformer block, there are two residual connections: one around attention, one around the MLP. If the input to the block is `x`, write out the computation flow showing both residual connections and both layer norms (using pre-norm style).
x = x+MLP(x) or x = x+attention(x)

---

## Putting It All Together

**Q18.** A GPT-style model stacks N transformer blocks. From input token IDs to output logits, list the full sequence of operations. (Embeddings → ... → logits)
embeddings -> add positional encodings -> attention block (multihead attention +  residual + layer norm + feed forward + residual + layer norm) repeat 6 times -> final output projection = logits

**Q19.** During training on language modeling, the model predicts the *next* token at every position. If the input is `[The, cat, sat, on]`, what are the targets? How does the loss work across the full sequence?

The targets are the actual next token within it's training sets. The loss is an average of log loss for all the tokens marked as targets. 

**Q20.** At inference time (text generation), we have a prompt and want to generate new tokens one at a time. Describe the autoregressive generation loop. Why can't we generate all tokens in parallel like we process them during training?
If we generate all the tokens paralelly it might not be consistent (i.e. may not read like an actual sentence) as we didn't impose dependencies between these tokens other than the fact that the prompt is the same prompt. 
