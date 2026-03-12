# Phase 0: Prerequisite Check — Questions

Answer these from memory / reasoning. No peeking at `pytorch_foundations.py`.
After you've written your answers, we'll review them together.

---

## Python Fundamentals

**Q1.** What is the difference between a Python list and a PyTorch tensor? Why can't you just use lists to train a neural network?

list does not support gpu and autograph, so we can not use list to do gradient descent.

> **Review note:** Correct on GPU and autograd, but missing the most fundamental reason: tensors support fast, vectorized math on contiguous memory. Lists are Python objects with per-element overhead — a matrix multiply on lists would be orders of magnitude slower even on CPU. Autograd is important but secondary to raw computation speed.

**Q2.** In the `TinyModel` class, `super().__init__()` is called in `__init__`. What does this do, and what would break if you removed it?

this is the __init__() of it's parent class, in this case nn.Modules. If removed, then TinyModel might not behave like a nn.Module class as some attributes are not initialized.

> **Review note:** Be more specific on "what breaks." `nn.Module.__init__()` initializes internal dicts (`_parameters`, `_modules`, `_buffers`). Without it, `self.linear = nn.Linear(...)` won't register the layer — so `model.parameters()` returns nothing, `model.to(device)` misses those layers, and the optimizer has nothing to optimize. It doesn't "might not behave" — it **will** break.

**Q3.** What does `for name, param in model.named_parameters()` give you that `model.parameters()` alone does not? When is this useful?
It gives you not jut the parameters tensor but also the name, easy to inspect / debug.

> **Review note:** Also useful for **freezing specific layers** (e.g., `if "embedding" in name: param.requires_grad = False`) and for **loading/saving state dicts** where parameter names are the keys.

---

## Linear Algebra

**Q4.** You have matrix A of shape `(2, 3)` and matrix B of shape `(3, 5)`. What is the shape of `A @ B`? What must be true about the inner dimensions for matrix multiplication to work?
(2,5). Not sure what's inner dimensions, if you mean when multiplying shape (a,b) and shape (c,d), then the rule is b==c. 

**Q5.** In attention, we compute `Q @ K.T`. If Q is shape `(batch, seq_len, d_model)` and K is the same shape, what shape is `K.T` (transposing the last two dims), and what is the resulting shape of `Q @ K.T`? What does each element in that output matrix represent?
Shape of K.T is (batch, d_model, seq_len), and resulting shape of Q @ (K.T) is (batch, seq_len, seq_len). The (b, i, j) element represents the attention score of the j th element w.r.t. the i th element, in the b th batch. 

**Q6.** What does softmax do, mathematically? Given the vector `[2.0, 1.0, 0.1]`, which element gets the highest probability, and why does softmax preserve the ordering while making values sum to 1?

\eta -> exp(\eta) / \sum exp(\eta). Preverses ordering because exp() is monotonic increasing, sums to 1 because the normalizing factor \sum exp(\eta).

**Q7.** What is a dot product? If vectors `a = [1, 2, 3]` and `b = [4, 5, 6]`, what is `a · b`? How does this relate to measuring similarity between two vectors?

geometrically a \cdot b = |a||b|cos(\theta) where \theta is the angle between a, b as a n-dimensional (here 3 dimensional) vector

> **Review note:** You gave the geometric definition but didn't compute the answer: `1*4 + 2*5 + 3*6 = 32`. For similarity: when vectors point in similar directions, `cos(θ) ≈ 1`, so the dot product is large. When orthogonal, it's 0. This is exactly why attention uses dot products — to measure how "related" two token representations are.

---

## Basic Calculus / Gradients

**Q8.** If `y = x² + 2x + 1`, what is `dy/dx`? At `x = 3`, what is the gradient value? (This was in the file — can you derive it rather than recall it?)
dy/dx = 2x+2 = 8 (if x = 3)

**Q9.** In the training loop, we call `loss.backward()`. In plain English, what does this compute and where does it store the results? 

It computes the gradients of all the tensors involved, and stores the result in .grad

> **Review note:** Not "all tensors involved" — only those with `requires_grad=True`. Intermediate tensors in the computation graph get gradients computed transiently, but only **leaf tensors** (like model parameters) have `.grad` populated.

**Q10.** Why do we call `optimizer.zero_grad()` before (or after) each training step? What goes wrong if we forget it?

To reset the gradients, otherwise gradients will accumulate by default, i.e. the gradient of last batch will remain in this batch's training. 

**Q11.** What is the purpose of gradient clipping (`clip_grad_norm_`)? What problem does it prevent, and when does that problem typically occur?

To prevent gradient from exploding, probably due to some outliers / not clean data.

> **Review note: Cause is wrong.** Exploding gradients are **not** primarily caused by outliers/dirty data. The main cause is **long chains of multiplications in deep networks or long sequences** (especially RNNs). When gradients are backpropagated through many layers/timesteps, repeated multiplication can cause them to grow exponentially. Can also happen with certain loss spikes, but "dirty data" is misleading as a primary explanation.
---

## PyTorch Mechanics

**Q12.** What is the difference between `view()` and `reshape()`? When would `view()` fail but `reshape()` would succeed?

view() does not create a new copy, but it requires continuous memory. reshape() might create a new copy (not sure if it's related to contiguous memory at all)

> **Review note:** Your intuition was right — it **is** about contiguous memory. `view()` requires the tensor to be contiguous in memory; if it's not (e.g., after `transpose()`), `view()` fails. `reshape()` returns a view if possible, or copies the data into contiguous memory if needed. The term is "contiguous" not "continuous."

**Q13.** Explain broadcasting: if you add a tensor of shape `(3, 4)` to a tensor of shape `(4,)`, what happens? What is the shape of the result?

it first broadcast (4,) to (3,4) (i.e. creating 3 identical rows, each equals to the original tensor (view (4,) as a row vector of shape (1,4))), then add it to (3,4). 

**Q14.** What does `nn.Embedding(1000, 64)` actually store internally? If you pass it `torch.tensor([5, 10, 23])`, what operation does it perform under the hood?

It stores a tensor of shape (1000,64); If we pass torch.tensor([5, 10, 23]), it returns a tensor of shape (3,64), with firt row the first row of the shape (1000,64) tensor etc.

> **Review note: Indexing is wrong.** It returns rows 5, 10, and 23 — not rows 0, 1, 2. It's a **table lookup**: `weight[[5, 10, 23]]`, so the first output row is `weight[5]`, not `weight[0]`.

**Q15.** Why do we need `model.train()` and `model.eval()`? Name at least one layer that behaves differently between these two modes.

Dropout layer behaves differently between two modes, as dropout layer will get turned off during eval phase.

> **Review note:** Correct. BatchNorm is the other major example — in train mode it uses per-batch statistics; in eval mode it uses running averages accumulated during training.

**Q16.** What does `torch.no_grad()` do and why is it used during inference? What two benefits does it provide?

It reset gradient for next step, to prevent gradients from accumulating. (I can only think of this one benefit)

> **Review note: This is wrong.** `torch.no_grad()` does **NOT** reset gradients — that's `zero_grad()`. `torch.no_grad()` **disables gradient computation entirely**: PyTorch won't build the computation graph inside the context manager. Two benefits: (1) **saves memory** — no computation graph stored, (2) **faster computation** — no autograd tracking overhead. This is a distinct operation from `zero_grad()` — don't confuse them.
---

## Putting It Together

**Q17.** In the training loop, the sequence of operations is: `forward → loss → backward → clip → step → zero_grad`. Explain what each step does in one sentence.

forward = model compute values under the new batch of inputs 
loss = compute the loss of this batch of inputs (i.e. measure the distance between predictions and targets)
backward = compute the gradients of each parameter, under this loss; in other words, compute the optimal direction to update the parameters to reduce this loss

> **Review note on "optimal direction":** The gradient is the direction of steepest **ascent** of the loss. The optimizer steps in the **negative** gradient direction (descent). "Optimal" is misleading — the gradient is just the local slope. How the update is actually computed depends on the optimizer (SGD vs Adam, etc.).
clip = clip the gradient, prevent it from exploding 
step = update parameters based on the clipped gradient and learning_rate 
zero_grad = clear gradients so it does not accumulate 

**Q18.** When computing cross-entropy loss for language modeling, why do we reshape logits from `(batch, seq_len, vocab_size)` to `(batch * seq_len, vocab_size)`? What does `F.cross_entropy` expect as input shapes?
F.cross_entropy expects input shapes (F, vocab_size) and (F,), therefore we need to reshape it. 

**Q19.** Explain the causal mask: what does `torch.tril()` produce, why do we fill the upper triangle with `-inf`, and what happens to those `-inf` values after softmax?
`torch.tril()` produces an lower triangular matrixs with 0 on the upper right corner and 1 otherwise. Because -inf becomes 0 after softmax, this is useful in transformer as we want the attention scores of the j th token w.r.t. to the ith token to be 0, when i > j

> **Review note: Inequality is backwards.** The mask prevents token `i` from attending to token `j` when **j > i** (future tokens). The lower triangle keeps positions where `j ≤ i` (current and past tokens). Your "when i > j" is flipped.

**Q20.** You have a model on GPU and a tensor on CPU. You run `model(tensor)`. What happens? How do you fix it?

It will crash, as it is required that all the tensors are on the same device.

> **Review note:** Correct. Fix with `tensor = tensor.to(device)` where `device` matches the model (e.g., `"cuda"`), or `tensor = tensor.cuda()`.
