# Personal LLM - Train a Language Model from Scratch

A hands-on learning project to build and train a GPT-style language model from scratch in PyTorch.

## Project Goal

Understand LLMs deeply by implementing every component by hand — tokenization, transformer architecture, training loop, and text generation.

## Plan

### Phase 0: Prerequisites (1-2 weeks)
- Solid Python skills
- Linear algebra basics (matrix multiply, dot products, softmax)
- Basic calculus (gradients, chain rule conceptually)
- PyTorch fundamentals (tensors, autograd, `nn.Module`)

### Phase 1: Understand the Transformer (1-2 weeks)
- Read "Attention Is All You Need" (decoder side)
- Watch Karpathy's Zero to Hero series
- Implement each component in a notebook:
  - Token embeddings + positional embeddings
  - Scaled dot-product attention
  - Multi-head attention with causal masking
  - Feed-forward network (MLP block)
  - Layer normalization
  - Residual connections

### Phase 2: Build a Tiny GPT (~1 week)
- Implement full GPT model in ~300-600 lines of PyTorch
- Follow karpathy/build-nanogpt commit-by-commit
- Train on Shakespeare (~1MB text) locally
- Get text generation working (greedy + temperature sampling)

### Phase 3: Build the Training Pipeline (1-2 weeks)
- Implement BPE tokenizer (or use tiktoken/HuggingFace initially)
- Build data loader for larger datasets
- Implement LR scheduling (warmup + cosine decay)
- Add gradient clipping, weight decay
- Add logging (loss curves, generated samples)

### Phase 4: Scale Up (2-4 weeks)
- Train 10M-125M param model on real data (FineWeb, OpenWebText)
- Rent cloud GPU (A100) or use own RTX 3090/4090
- Implement mixed-precision training (fp16/bf16)
- Optionally: distributed training (DDP)

### Phase 5 (Optional): Post-Training
- Supervised fine-tuning (SFT) on instruction data
- Explore RLHF / DPO concepts
- Build a simple chat interface

## Key Resources

- [Karpathy's Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [build-nanogpt](https://github.com/karpathy/build-nanogpt)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [LLMs from Scratch (rasbt)](https://github.com/rasbt/LLMs-from-scratch)
- Paper: "Attention Is All You Need" (Vaswani et al., 2017)
- Paper: GPT-2 (Radford et al., 2019)

## Datasets

- **Tiny/toy**: Shakespeare (~1MB) — for debugging and first runs
- **Small**: OpenWebText (~38GB) — GPT-2 scale
- **Medium**: FineWeb, DCLM — curated web data
- **Lists**: [LLMDataHub](https://github.com/Zjh-819/LLMDataHub), [mlabonne/llm-datasets](https://github.com/mlabonne/llm-datasets)

## Budget

| Path | Cost | What you get |
|---|---|---|
| A: Learning only (CPU/Colab) | $0-10/mo | Tiny model on Shakespeare, full understanding |
| B: Cloud GPU training | $100-500 | 125M param model, GPT-2 small scale |
| C: Own GPU (RTX 3090/4090) | $800-1,600 | Long-term experimentation |

## Conventions

- All code in PyTorch, no high-level LLM wrapper libraries
- Prioritize readability and learning over performance
- Start absurdly small (2 layers, 64 dims), verify, then scale
- Always overfit a single batch first before full training runs

## Task Log

When working on a task, maintain a running log at `.claude/task-log.md`.
- At the start of a task, write a one-line summary of the goal.
- After each significant step (file created, test run, decision made), append a short line.
- If you're waiting on something or hit a blocker, note it.
- Keep entries terse — one line each, timestamp not needed.
