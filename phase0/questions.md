# Phase 0: Prerequisite Check — Questions

Answer these from memory / reasoning. No peeking at `pytorch_foundations.py`.
After you've written your answers, we'll review them together.

---

## Python Fundamentals

**Q1.** What is the difference between a Python list and a PyTorch tensor? Why can't you just use lists to train a neural network?

list does not support gpu and autograph, so we can not use list to do gradient descent. 

**Q2.** In the `TinyModel` class, `super().__init__()` is called in `__init__`. What does this do, and what would break if you removed it?

this is the __init__() of it's parent class, in this case nn.Modules. If removed, then TinyModel might not behave like a nn.Module class as some attributes are not initialized. 

**Q3.** What does `for name, param in model.named_parameters()` give you that `model.parameters()` alone does not? When is this useful?
It gives you not jut the parameters tensor but also the name, easy to inspect / debug. 

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

---

## Basic Calculus / Gradients

**Q8.** If `y = x² + 2x + 1`, what is `dy/dx`? At `x = 3`, what is the gradient value? (This was in the file — can you derive it rather than recall it?)
dy/dx = 2x+2 = 8 (if x = 3)

**Q9.** In the training loop, we call `loss.backward()`. In plain English, what does this compute and where does it store the results? 

It computes the gradients of all the tensors involved, and stores the result in .grad

**Q10.** Why do we call `optimizer.zero_grad()` before (or after) each training step? What goes wrong if we forget it?

To reset the gradients, otherwise gradients will accumulate by default, i.e. the gradient of last batch will remain in this batch's training. 

**Q11.** What is the purpose of gradient clipping (`clip_grad_norm_`)? What problem does it prevent, and when does that problem typically occur?

To prevent gradient from exploding, probably due to some outliers / not clean data. 
---

## PyTorch Mechanics

**Q12.** What is the difference between `view()` and `reshape()`? When would `view()` fail but `reshape()` would succeed?

view() does not create a new copy, but it requires continuous memory. reshape() might create a new copy (not sure if it's related to contiguous memory at all)

**Q13.** Explain broadcasting: if you add a tensor of shape `(3, 4)` to a tensor of shape `(4,)`, what happens? What is the shape of the result?

it first broadcast (4,) to (3,4) (i.e. creating 3 identical rows, each equals to the original tensor (view (4,) as a row vector of shape (1,4))), then add it to (3,4). 

**Q14.** What does `nn.Embedding(1000, 64)` actually store internally? If you pass it `torch.tensor([5, 10, 23])`, what operation does it perform under the hood?

It stores a tensor of shape (1000,64); If we pass torch.tensor([5, 10, 23]), it returns a tensor of shape (3,64), with firt row the first row of the shape (1000,64) tensor etc. 

**Q15.** Why do we need `model.train()` and `model.eval()`? Name at least one layer that behaves differently between these two modes.

Dropout layer behaves differently between two modes, as dropout layer will get turned off during eval phase. 

**Q16.** What does `torch.no_grad()` do and why is it used during inference? What two benefits does it provide?

It reset gradient for next step, to prevent gradients from accumulating. (I can only think of this one benefit)
---

## Putting It Together

**Q17.** In the training loop, the sequence of operations is: `forward → loss → backward → clip → step → zero_grad`. Explain what each step does in one sentence.

forward = model compute values under the new batch of inputs 
loss = compute the loss of this batch of inputs (i.e. measure the distance between predictions and targets)
backward = compute the gradients of each parameter, under this loss; in other words, compute the optimal direction to update the parameters to reduce this loss
clip = clip the gradient, prevent it from exploding 
step = update parameters based on the clipped gradient and learning_rate 
zero_grad = clear gradients so it does not accumulate 

**Q18.** When computing cross-entropy loss for language modeling, why do we reshape logits from `(batch, seq_len, vocab_size)` to `(batch * seq_len, vocab_size)`? What does `F.cross_entropy` expect as input shapes?
F.cross_entropy expects input shapes (F, vocab_size) and (F,), therefore we need to reshape it. 

**Q19.** Explain the causal mask: what does `torch.tril()` produce, why do we fill the upper triangle with `-inf`, and what happens to those `-inf` values after softmax?
`torch.tril()` produces an lower triangular matrixs with 0 on the upper right corner and 1 otherwise. Because -inf becomes 0 after softmax, this is useful in transformer as we want the attention scores of the j th token w.r.t. to the ith token to be 0, when i > j

**Q20.** You have a model on GPU and a tensor on CPU. You run `model(tensor)`. What happens? How do you fix it?

It will crash, as it is required that all the tensors are on the same device. 
