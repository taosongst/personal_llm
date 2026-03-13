# Task Log

## Phase 0: PyTorch Foundations
- Goal: Get comfortable with PyTorch basics needed for LLM building
- Created phase0/pytorch_foundations.py — covers tensors, autograd, nn.Module, training loop, masking, GPU usage
- Created phase0/questions.md — 20 questions covering Python, linear algebra, calculus, PyTorch mechanics
- Added #Answer blocks to 8 inline questions in pytorch_foundations.py
- Reviewed 8 exercise answers — 7/8 correct, corrected misconception on nn.Embedding (it's indexing, not hashing)
- Reviewed user's answers to all 20 questions in questions.md
- Added review notes as blockquotes: 1 wrong (Q16: no_grad vs zero_grad), 1 backwards (Q19: mask inequality), several incomplete
## Phase 1: Understand the Transformer
- Goal: Deep understanding of transformer architecture before implementing
- Created phase1/questions.md — 20 questions covering embeddings, attention, multi-head, masking, MLP, LayerNorm, residuals, and full architecture
- Created phase1/attention-paper-summary.md — thorough section-by-section summary of "Attention Is All You Need" with ~40 guiding questions
- Created phase1/concept-notes.md — documented Q&A discussions: encoder vs decoder, causal masking, dimensionality flow, positional encoding intuition, RoPE & ALiBi
- Reviewed phase1/questions.md answers: 8 solid, 6 need depth, 6 wrong/significantly off. Key issues: Q/K swapped (Q4), multi-head reshape (Q8), W_O purpose (Q9), GELU description (Q13), incomplete block flow (Q17), wrong autoregressive reasoning (Q20)
## Phase 2: Build a Tiny GPT
- Goal: Implement a tiny GPT from scratch and train on Shakespeare
- Downloaded Shakespeare dataset (~1MB) to phase2/data/input.txt
- Created phase2/train_gpt.py guided skeleton — 6 steps with TODOs, shape comments, and verification checks
- Config: 4 layers, 4 heads, 128 dims, ~800K params, char-level tokenizer
