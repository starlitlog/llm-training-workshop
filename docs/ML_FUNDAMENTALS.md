# ML Fundamentals Glossary

A quick reference for key machine learning concepts, especially for LLM training.

---

## Training Concepts

### Gradient Descent

The core optimization algorithm. To minimize loss:
1. Compute gradient (slope) of loss with respect to each weight
2. Move weights in opposite direction of gradient
3. Repeat

```
w_new = w_old - learning_rate × gradient
```

### Gradient Norm

The magnitude (length) of the gradient vector across all parameters:
```
grad_norm = sqrt(g1² + g2² + ... + gN²)
```

| Grad Norm | Meaning |
|-----------|---------|
| ~1-2 | Healthy, stable training |
| → 0 | Vanishing gradients (stopped learning) |
| → 100+ | Exploding gradients (unstable) |

### Backpropagation

How gradients are computed efficiently. Uses the **chain rule** to propagate error backwards through layers:

```
Layer 3 → Layer 2 → Layer 1
   ↑ error    ↑         ↑
   └──────────┴─────────┘
   Chain rule computes each layer's contribution
```

Without backprop, you'd have no idea how to update deep layers.

### Stochastic Gradient Descent (SGD)

Basic optimizer - update weights using gradient from a random batch:
```
w = w - lr × gradient
```

Problems:
- Same learning rate for all weights
- Noisy gradients cause jittery updates

### Adam Optimizer

Improved optimizer combining two ideas:

1. **Momentum**: Smooth out noisy gradients by averaging
2. **Adaptive LR**: Different learning rate per weight

```python
momentum = 0.9 × momentum + 0.1 × gradient      # Smooths noise
velocity = running_avg(gradient²)               # Tracks per-weight variance
update = momentum / sqrt(velocity)              # Adaptive step size
```

Adam converges faster and more reliably than vanilla SGD.

---

## Loss Functions

### Cross-Entropy Loss

The standard loss for language models. Measures "how surprised is the model by the correct answer?"

```
Loss = -log(P(correct_token))
```

| Model Confidence | Loss |
|------------------|------|
| P(correct) = 0.9 | 0.1 (low - good!) |
| P(correct) = 0.5 | 0.7 (medium) |
| P(correct) = 0.01 | 4.6 (high - bad!) |

### Perplexity

Intuitive measure derived from cross-entropy:
```
Perplexity = e^(cross_entropy_loss)
```

Interpretation: "How many tokens is the model equally confused between?"

| Perplexity | Meaning |
|------------|---------|
| 1 | Perfect certainty |
| 10 | Choosing between ~10 options |
| 100 | Very confused |
| 50000 | Random guessing (vocab size) |

---

## Regularization

### Weight Decay (L2 Regularization)

Prevents overfitting by shrinking weights toward zero each step:
```
w = w × (1 - weight_decay) - lr × gradient
        ↑
        Gentle pressure toward smaller weights
```

- Large weights → model memorizing specific examples
- Small weights → model learning general patterns
- Typical value: 0.01

### Dropout

Randomly zero out neurons during training (e.g., 10% dropout). Forces network to not rely on any single neuron. Disabled during inference.

### Gradient Checkpointing

Memory optimization: don't store all activations, recompute them during backward pass.

Trade-off: ~30% slower, ~40% less memory.

---

## Learning Rate

### Why Decay Learning Rate?

- **Early training**: Large LR for fast progress
- **Late training**: Small LR for precise convergence

Fixed LR either:
- Too small → slow start
- Too large → overshoots at the end

### Cosine Schedule

```
Linear decay:  ████████░░░░░░░░░░  (steady decline)
Cosine decay:  ██████████████░░░░  (slow-fast-slow)
```

Cosine is gentler at start and end, often finds better minima.

### Warmup

Gradually increase LR from 0 at the start (e.g., first 3% of training). Prevents early instability when weights are random.

---

## Tokenization

### The Problem

- **Word-level**: "unhappiness" = 1 token, "unhappiest" = UNKNOWN
- **Character-level**: No unknowns, but sequences too long

### BPE (Byte Pair Encoding)

Split into **frequent subwords**:
```
"unhappiness" → ["un", "happi", "ness"]
"unhappiest"  → ["un", "happi", "est"]
```

Benefits:
- Fixed vocabulary size (~32K-100K)
- No unknown words ever
- Shares knowledge across word forms

### Tokens vs Words

Rough rule: **1 token ≈ 0.75 words** (or ~4 characters)

```
"Hello world" → [15496, 995]  (2 tokens)
"transformers" → [9003, 388]   (2 tokens)
```

---

## Architecture Concepts

### Embeddings

Words → Dense vectors where meaning = position in space

```
king - man + woman ≈ queen
```

Each token becomes a high-dimensional vector (e.g., 4096 dims). The model learns these positions during training.

### Softmax

Converts raw scores (logits) to probabilities that sum to 1:
```
P(token_i) = e^(logit_i) / Σ e^(logit_j)
```

### Temperature

Scaling factor applied before softmax:
```
P(token_i) = e^(logit_i / T) / Σ e^(logit_j / T)
```

| Temperature | Effect |
|-------------|--------|
| T → 0 | Always pick highest (deterministic) |
| T = 1 | Normal sampling |
| T = 2 | Flatter distribution (creative/random) |

### Residual Connections

Every transformer block: `output = layer(x) + x`

The `+ x` creates a skip connection. Without it:
- Gradients must survive through every layer
- 40 layers × small shrinkage = vanishing gradients

With residuals, gradients have a "highway" straight back.

### Layer Normalization

Normalize activations to zero mean, unit variance within each layer. Stabilizes training by preventing internal covariate shift.

### Positional Encoding

Attention treats input as a "bag of tokens" - no inherent order. Solution: add position information to embeddings:

```
input = token_embedding + position_embedding
```

Each position (0, 1, 2, ...) has a unique vector. The model learns that "dog bites man" ≠ "man bites dog" because the same tokens have different position vectors.

### Causal Masking

During training, the model must not see future tokens. Why?

1. **Prevents cheating**: If it can see the answer, it just copies instead of learning patterns
2. **Matches inference**: At generation time, future tokens don't exist yet

```
Training:  "The capital of France is [Paris]"
                                       ↑
           Model can only see tokens BEFORE this position
```

Implemented as a triangular mask in the attention matrix - each position can only attend to earlier positions.

---

## Inference & Generation

### KV Cache

During generation, naive approach recomputes Key and Value matrices for ALL previous tokens at each step. Wasteful!

**KV cache solution**: Store K and V from previous steps, only compute for new token:

```
Step 1: Compute K,V for token 1      → cache [K1, V1]
Step 2: Compute K,V for token 2 only → cache [K1, K2], [V1, V2]
Step 3: Compute K,V for token 3 only → cache [K1, K2, K3], [V1, V2, V3]
```

Result: **10x+ speedup** for long sequences. Memory trade-off: cache grows with sequence length.

### Top-k Sampling

Pick only from the top k most likely tokens:
```
Top-k=10: Always sample from exactly 10 highest probability tokens
```

Problem: k is fixed regardless of model confidence.

### Top-p (Nucleus) Sampling

Pick from smallest set of tokens whose probabilities sum to p:
```
Top-p=0.9: Sample from tokens that together have 90% probability
```

**Dynamic** - adapts to confidence:
- Model confident → maybe 2-3 tokens reach 90%
- Model uncertain → might need 50 tokens to reach 90%

Temperature + top-p together control creativity.

### Greedy vs Sampling

| Method | How it works | Use case |
|--------|--------------|----------|
| Greedy | Always pick highest probability | Deterministic, factual |
| Sampling | Random pick weighted by probability | Creative, varied |
| Beam search | Track top-n candidates | Translation, structured output |

---

## Memory & Precision

### bfloat16 vs float16 vs float32

| Type | Bits | Range | Precision |
|------|------|-------|-----------|
| float32 | 32 | High | High |
| float16 | 16 | Limited | Medium |
| bfloat16 | 16 | High (like f32) | Lower |

bfloat16 is preferred for training: same range as float32, half the memory.

### Quantization

Reduce precision to save memory:

| Precision | Memory per param | Quality |
|-----------|------------------|---------|
| float16 | 2 bytes | Full |
| int8 | 1 byte | Slight loss |
| int4 | 0.5 bytes | More loss |

QLoRA uses 4-bit base model + full precision adapters.

---

## Training Dynamics

### Batch Size vs Gradient Accumulation

```
effective_batch_size = batch_size × gradient_accumulation_steps
```

Larger effective batch = more stable gradients, but uses more memory.

Gradient accumulation: simulate larger batch by accumulating gradients over multiple forward passes before updating.

### Epochs

One epoch = one complete pass through all training data.

For small datasets: multiple epochs (3-5)
For large datasets: often < 1 epoch

### Overfitting vs Underfitting

| Problem | Symptom | Fix |
|---------|---------|-----|
| Overfitting | Train loss low, eval loss high | More data, regularization, dropout |
| Underfitting | Both losses high | Larger model, more epochs, higher LR |

---

## Quick Reference

| Concept | One-liner |
|---------|-----------|
| Gradient | Direction of steepest loss increase |
| Learning rate | Step size for weight updates |
| Batch size | Samples per gradient computation |
| Epoch | One pass through all data |
| Loss | How wrong the model is |
| Perplexity | e^loss - "how many choices confused between" |
| Adam | SGD + momentum + adaptive LR |
| Weight decay | Shrink weights to prevent overfitting |
| BPE | Subword tokenization for fixed vocab |
| Embeddings | Words as vectors in meaning-space |
| Residuals | Skip connections for gradient flow |
| Temperature | Softmax sharpness control |
