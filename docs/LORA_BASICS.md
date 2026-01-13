# LoRA Basics

A practical guide to Low-Rank Adaptation for fine-tuning large language models.

## Foundation: From Perceptrons to Transformers

Before diving into LoRA, it helps to understand the evolution of neural networks and where LoRA fits in.

### The Perceptron (1950s)

The simplest neural unit - a single "neuron":

```
inputs: [x1, x2, x3]
weights: [w1, w2, w3]
output = activation(w1*x1 + w2*x2 + w3*x3 + bias)
```

- Takes inputs, multiplies by weights, sums them up
- Passes through an activation function (step, sigmoid, ReLU)
- **Learning** = adjusting the weights to minimize error

### Neural Networks (1980s-2000s)

Stack perceptrons into layers:

```
Input Layer → Hidden Layer(s) → Output Layer
    [x]    →      [h]        →     [y]
```

**MLP (Multi-Layer Perceptron)**:
- Multiple layers of neurons
- Each layer is a matrix multiplication: `output = activation(W × input + bias)`
- **Backpropagation**: compute gradients, update weights layer by layer
- More layers = can learn more complex patterns

The "deep" in deep learning = many hidden layers.

### The Problem with Sequences

Traditional NNs process fixed-size inputs. But language is sequential - words depend on context:

> "The bank by the river" vs "The bank approved the loan"

"Bank" means different things based on surrounding words.

### Recurrent Neural Networks (RNNs)

Process sequences by maintaining "memory":

```
h[t] = f(h[t-1], x[t])  # Current state depends on previous state + current input
```

Problems:
- **Vanishing gradients**: Long sequences lose information
- **Sequential processing**: Can't parallelize, slow training

### The Transformer (2017)

"Attention Is All You Need" - the architecture behind GPT, BERT, LLaMA, Gemma.

Key insight: **Don't process sequentially. Let every token "look at" every other token simultaneously.**

```
Input: "The cat sat on the mat"
        ↓
   [Attention] ← Every word attends to every other word
        ↓
      [MLP]    ← Process each position
        ↓
      Output
```

### Attention Mechanism

The core innovation. For each token, compute:

1. **Query (Q)**: "What am I looking for?"
2. **Key (K)**: "What do I contain?"
3. **Value (V)**: "What information do I provide?"

```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d)) × V
```

In practice:
```
Q = input × W_q  (query projection)
K = input × W_k  (key projection)
V = input × W_v  (value projection)
output = attention_result × W_o  (output projection)
```

Those `W_q`, `W_k`, `W_v`, `W_o` matrices are the **attention weights** - and they're what LoRA targets with `q_proj`, `k_proj`, `v_proj`, `o_proj`.

### Multi-Head Attention

Run multiple attention operations in parallel ("heads"), each learning different patterns:
- Head 1 might learn syntax
- Head 2 might learn entity relationships
- Head 3 might learn sentiment

```python
# Typical config
num_attention_heads: 32  # 32 parallel attention operations
hidden_size: 4096        # Dimension of representations
head_dim: 128            # 4096 / 32 = 128 per head
```

### The Full Transformer Block

```
┌─────────────────────────────────────┐
│            Input Tokens             │
└──────────────┬──────────────────────┘
               ↓
┌──────────────────────────────────────┐
│    Multi-Head Self-Attention         │
│   (q_proj, k_proj, v_proj, o_proj)   │  ← LoRA targets these
└──────────────┬───────────────────────┘
               ↓
          [Add & Norm]  ← Residual connection
               ↓
┌──────────────────────────────────────┐
│         Feed-Forward MLP             │
│   (gate_proj, up_proj, down_proj)    │  ← LoRA targets these
└──────────────┬───────────────────────┘
               ↓
          [Add & Norm]  ← Residual connection
               ↓
┌──────────────────────────────────────┐
│           Output Tokens              │
└──────────────────────────────────────┘
```

A model like Gemma 27B has **~32-40 of these blocks** stacked.

### Why So Many Parameters?

Each block has:
- Attention projections: 4 matrices × (4096 × 4096) = ~67M params
- MLP: 3 matrices × (4096 × 16384) = ~200M params
- Total per block: ~270M parameters
- 32 blocks × 270M = **8.6B** just in the main layers

Plus embeddings, layer norms, etc. → 27B total.

### The Fine-Tuning Problem

To teach new knowledge:
- **Full fine-tuning**: Update all 27B parameters
- Needs ~108GB VRAM (fp16)
- Risk of "catastrophic forgetting"

**Enter LoRA**: Only update small adapter matrices, keep base model frozen.

---

## What is LoRA?

**LoRA (Low-Rank Adaptation)** is a technique that makes fine-tuning large models practical by training only a small number of additional parameters instead of the full model weights.

Instead of updating the massive weight matrices directly (which would require storing billions of parameters), LoRA adds small "adapter" matrices that learn the task-specific adjustments.

## How It Works

### The Core Idea

For a weight matrix `W` in the model:
- **Full fine-tuning**: Update all of `W` (billions of parameters)
- **LoRA**: Keep `W` frozen, add `A × B` where A and B are small matrices

```
Original:     output = W × input
With LoRA:    output = W × input + (A × B) × input
                       ↑ frozen    ↑ trainable
```

### Low-Rank Decomposition

If `W` is a `4096 × 4096` matrix (16M parameters), LoRA uses:
- `A`: `4096 × 64` matrix (262K parameters)
- `B`: `64 × 4096` matrix (262K parameters)

The product `A × B` produces a `4096 × 4096` update, but we only train 524K parameters instead of 16M - a **97% reduction**.

## Key Parameters

### `lora_r` (Rank)

The **rank** determines the dimensionality of the A and B matrices.

```yaml
lora_r: 64  # The "bottleneck" dimension
```

- **Higher rank** (64, 128) = more learning capacity, more parameters
- **Lower rank** (8, 16) = fewer parameters, may underfit complex tasks
- Think of it as: "How many dimensions of variation can the model learn?"

### `lora_alpha` (Scaling Factor)

Controls how much the LoRA update affects the output.

```yaml
lora_alpha: 128  # Scaling factor
```

The actual scaling applied is `alpha / r`. With `alpha=128` and `r=64`:
- Scaling = 128/64 = 2x
- The LoRA contribution is doubled

**Rule of thumb**: `alpha = 2 × r` is a common starting point.

### `lora_dropout`

Regularization to prevent overfitting.

```yaml
lora_dropout: 0.05  # 5% dropout on LoRA layers
```

### `lora_target_modules`

Which layers to apply LoRA to. In transformer models:

```yaml
lora_target_modules:
  - q_proj    # Query projection (attention)
  - k_proj    # Key projection (attention)
  - v_proj    # Value projection (attention)
  - o_proj    # Output projection (attention)
  - gate_proj # MLP gate
  - up_proj   # MLP up projection
  - down_proj # MLP down projection
```

**Attention layers** (`q_proj`, `k_proj`, `v_proj`, `o_proj`):
- Control how the model attends to different parts of the input
- Essential for learning new patterns

**MLP layers** (`gate_proj`, `up_proj`, `down_proj`):
- The "feed-forward" network in each transformer block
- Where much of the "knowledge" is stored

More target modules = more learning capacity but more parameters.

## Understanding the Architecture

### Transformer Block Refresher

Each transformer layer has:
1. **Self-Attention**: Queries, Keys, Values determine what to focus on
2. **MLP (Multi-Layer Perceptron)**: Feed-forward network that processes each position

```
Input
  ↓
[Self-Attention] ← q_proj, k_proj, v_proj, o_proj
  ↓
[MLP]            ← gate_proj, up_proj, down_proj
  ↓
Output
```

### Where Learning Happens

With LoRA rank 64 targeting all 7 modules across (say) 32 layers:
- Each adapter pair: ~500K parameters
- 7 modules × 32 layers × 500K = ~112M trainable parameters
- vs. 27B total parameters = **0.4% of the model**

But those 112M parameters affect the **entire computation** through the low-rank updates.

## QLoRA: Quantized LoRA

**QLoRA** combines LoRA with quantization for even more memory savings:

```yaml
load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
```

| Approach | Base Model | Adapters | Memory |
|----------|-----------|----------|--------|
| Full fine-tune | float16 | N/A | Very High |
| LoRA | float16 | float16 | High |
| QLoRA | 4-bit | float16 | Low |

### How QLoRA Works

1. Base model weights quantized to 4-bit (NormalFloat4)
2. LoRA adapters remain in full precision (float16/bfloat16)
3. During forward pass: dequantize → compute → quantize
4. Gradients only flow through the full-precision adapters

**Trade-off**: Slower training (dequantization overhead) but fits larger models in memory.

## Practical Tips

### Choosing Rank

| Task Complexity | Suggested Rank |
|----------------|----------------|
| Simple (style transfer) | 8-16 |
| Medium (domain adaptation) | 32-64 |
| Complex (new capabilities) | 64-128 |

### Memory Estimates (approximate)

For a 27B parameter model:
- Full fine-tune: ~108GB (fp16) - impossible on most GPUs
- LoRA (r=64): ~54GB base + ~500MB adapters
- QLoRA (r=64, 4-bit): ~14GB base + ~500MB adapters

### Common Configurations

**Conservative (small dataset)**:
```yaml
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
```

**Balanced (medium dataset)**:
```yaml
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
```

**Aggressive (large dataset)**:
```yaml
lora_r: 128
lora_alpha: 256
lora_dropout: 0.05
```

## Merging LoRA Weights

After training, you can merge LoRA weights into the base model:

```python
# Merge adapters into base model
merged_model = model.merge_and_unload()

# Save as standalone model
merged_model.save_pretrained("path/to/merged")
```

The merged model:
- No longer needs PEFT library to load
- Same size as original base model
- Inference speed identical to base model

## Hyperparameter Reference

### LoRA Parameters

| Parameter | Meaning | Typical Values | Pattern/Rule |
|-----------|---------|----------------|--------------|
| `lora_r` | Rank - dimensionality of low-rank matrices | 8, 16, 32, 64, 128 | Higher = more capacity, more params. Start with 32-64 |
| `lora_alpha` | Scaling factor for LoRA updates | 16, 32, 64, 128, 256 | Usually `alpha = 2 × r` or `alpha = r` |
| `lora_dropout` | Dropout on LoRA layers (regularization) | 0.0, 0.05, 0.1 | Higher for small datasets to prevent overfitting |
| `lora_target_modules` | Which layers to adapt | q,k,v,o_proj + MLP | More modules = more learning, more memory |

### Training Parameters

| Parameter | Meaning | Typical Values | Pattern/Rule |
|-----------|---------|----------------|--------------|
| `epochs` | Full passes through dataset | 1-5 | More epochs for small datasets; watch for overfitting |
| `batch_size` | Samples per forward pass | 1, 2, 4, 8 | Limited by GPU memory; larger = more stable gradients |
| `gradient_accumulation_steps` | Virtual batch size multiplier | 2, 4, 8, 16 | effective_batch = batch_size × grad_accum |
| `lr` (learning_rate) | Step size for weight updates | 1e-5 to 2e-4 | LoRA can use higher LR than full fine-tune |
| `warmup_ratio` | Fraction of steps for LR warmup | 0.03, 0.05, 0.1 | Gradual ramp-up prevents early instability |
| `weight_decay` | L2 regularization strength | 0.0, 0.01, 0.1 | Prevents overfitting; 0.01 is common default |
| `lr_scheduler_type` | How LR changes over training | linear, cosine, constant | Cosine often works best |
| `max_length` | Max sequence length (tokens) | 256, 512, 1024, 2048 | ~0.75 words per token; longer = more memory |

### Precision & Memory

| Parameter | Meaning | Typical Values | Pattern/Rule |
|-----------|---------|----------------|--------------|
| `bf16` | Use bfloat16 precision | true/false | Preferred on Ampere+ (A100, RTX 30xx+) |
| `fp16` | Use float16 precision | true/false | Use if bf16 not supported |
| `gradient_checkpointing` | Trade compute for memory | true/false | Enable if OOM; ~30% slower but ~40% less memory |
| `load_in_4bit` | QLoRA 4-bit quantization | true/false | Enables large models on small GPUs |
| `load_in_8bit` | 8-bit quantization | true/false | Middle ground: less compression, faster than 4-bit |

### QLoRA-Specific

| Parameter | Meaning | Typical Values | Pattern/Rule |
|-----------|---------|----------------|--------------|
| `bnb_4bit_quant_type` | Quantization algorithm | nf4, fp4 | NF4 (NormalFloat4) is usually better |
| `bnb_4bit_compute_dtype` | Dtype for computation | bfloat16, float16 | Match your precision setting |
| `bnb_4bit_use_double_quant` | Quantize the quantization constants | true/false | True saves ~0.4 bits/param extra |

### DataLoader

| Parameter | Meaning | Typical Values | Pattern/Rule |
|-----------|---------|----------------|--------------|
| `dataloader_num_workers` | Parallel data loading processes | 0, 2, 4, 8 | More = faster loading; 4 is usually good |
| `dataloader_prefetch_factor` | Batches to prefetch per worker | 2, 4 | Higher = more memory, faster throughput |

### Logging & Saving

| Parameter | Meaning | Typical Values | Pattern/Rule |
|-----------|---------|----------------|--------------|
| `logging_steps` | Log metrics every N steps | 10, 50, 100 | Lower = more visibility, slight overhead |
| `save_steps` | Save checkpoint every N steps | 50, 100, 500 | Balance between safety and disk space |
| `save_strategy` | When to save | steps, epoch | "epoch" for small datasets |
| `evaluation_strategy` | When to evaluate | steps, epoch | Match save_strategy usually |

### Common Patterns

**Small dataset (<1K samples)**:
```yaml
epochs: 3-5
lora_r: 16-32
lora_dropout: 0.1
lr: 1e-4
```

**Medium dataset (1K-10K samples)**:
```yaml
epochs: 2-3
lora_r: 64
lora_dropout: 0.05
lr: 2e-4
```

**Large dataset (>10K samples)**:
```yaml
epochs: 1-2
lora_r: 64-128
lora_dropout: 0.05
lr: 2e-4
```

**Memory constrained (< 24GB VRAM)**:
```yaml
load_in_4bit: true
gradient_checkpointing: true
batch_size: 1
gradient_accumulation_steps: 8
```

**Speed optimized (A100/H100)**:
```yaml
bf16: true
use_flash_attention: true
gradient_checkpointing: false
batch_size: 4-8
dataloader_num_workers: 4
```

---

## Summary

| Concept | What It Does |
|---------|--------------|
| LoRA | Trains small A×B matrices instead of full weights |
| Rank (r) | Dimensionality of adaptation (learning capacity) |
| Alpha | Scaling factor for LoRA contribution |
| Target modules | Which layers get LoRA adapters |
| QLoRA | 4-bit base model + full precision adapters |

LoRA makes it possible to fine-tune models like Gemma 27B on a single GPU by training <1% of parameters while still achieving strong task-specific performance.
