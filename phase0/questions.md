# Phase 0: Prerequisite Check — Questions

Answer these from memory / reasoning. No peeking at `pytorch_foundations.py`.
After you've written your answers, we'll review them together.

---

## Python Fundamentals

**Q1.** What is the difference between a Python list and a PyTorch tensor? Why can't you just use lists to train a neural network?

**Q2.** In the `TinyModel` class, `super().__init__()` is called in `__init__`. What does this do, and what would break if you removed it?

**Q3.** What does `for name, param in model.named_parameters()` give you that `model.parameters()` alone does not? When is this useful?

---

## Linear Algebra

**Q4.** You have matrix A of shape `(2, 3)` and matrix B of shape `(3, 5)`. What is the shape of `A @ B`? What must be true about the inner dimensions for matrix multiplication to work?

**Q5.** In attention, we compute `Q @ K.T`. If Q is shape `(batch, seq_len, d_model)` and K is the same shape, what shape is `K.T` (transposing the last two dims), and what is the resulting shape of `Q @ K.T`? What does each element in that output matrix represent?

**Q6.** What does softmax do, mathematically? Given the vector `[2.0, 1.0, 0.1]`, which element gets the highest probability, and why does softmax preserve the ordering while making values sum to 1?

**Q7.** What is a dot product? If vectors `a = [1, 2, 3]` and `b = [4, 5, 6]`, what is `a · b`? How does this relate to measuring similarity between two vectors?

---

## Basic Calculus / Gradients

**Q8.** If `y = x² + 2x + 1`, what is `dy/dx`? At `x = 3`, what is the gradient value? (This was in the file — can you derive it rather than recall it?)

**Q9.** In the training loop, we call `loss.backward()`. In plain English, what does this compute and where does it store the results?

**Q10.** Why do we call `optimizer.zero_grad()` before (or after) each training step? What goes wrong if we forget it?

**Q11.** What is the purpose of gradient clipping (`clip_grad_norm_`)? What problem does it prevent, and when does that problem typically occur?

---

## PyTorch Mechanics

**Q12.** What is the difference between `view()` and `reshape()`? When would `view()` fail but `reshape()` would succeed?

**Q13.** Explain broadcasting: if you add a tensor of shape `(3, 4)` to a tensor of shape `(4,)`, what happens? What is the shape of the result?

**Q14.** What does `nn.Embedding(1000, 64)` actually store internally? If you pass it `torch.tensor([5, 10, 23])`, what operation does it perform under the hood?

**Q15.** Why do we need `model.train()` and `model.eval()`? Name at least one layer that behaves differently between these two modes.

**Q16.** What does `torch.no_grad()` do and why is it used during inference? What two benefits does it provide?

---

## Putting It Together

**Q17.** In the training loop, the sequence of operations is: `forward → loss → backward → clip → step → zero_grad`. Explain what each step does in one sentence.

**Q18.** When computing cross-entropy loss for language modeling, why do we reshape logits from `(batch, seq_len, vocab_size)` to `(batch * seq_len, vocab_size)`? What does `F.cross_entropy` expect as input shapes?

**Q19.** Explain the causal mask: what does `torch.tril()` produce, why do we fill the upper triangle with `-inf`, and what happens to those `-inf` values after softmax?

**Q20.** You have a model on GPU and a tensor on CPU. You run `model(tensor)`. What happens? How do you fix it?
