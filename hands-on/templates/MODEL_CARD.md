---
license: mit
base_model: meta-llama/Llama-3.2-3B
tags:
  - llama
  - llama-3.2
  - fine-tuned
  - lora
  - peft
  - dsl
  - gridscript
  - domain-specific-language
language:
  - en
pipeline_tag: text-generation
---

# GridScript™ DSL Expert - Fine-Tuned Llama 3.2 3B

This model is a fine-tuned version of [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) using LoRA (Low-Rank Adaptation) for GridScript™ domain-specific language expertise.

## Model Details

- **Base Model:** meta-llama/Llama-3.2-3B
- **Fine-tuning Method:** LoRA (PEFT)
- **Language:** English
- **Domain:** GridScript™ DSL for multidimensional data modeling
- **Training Data:** 1,028 prompt-completion pairs

### Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 64 |
| LoRA Alpha | 128 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning Rate | 3e-4 |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.03 |
| Epochs | 5 |
| Batch Size | 8 |
| Gradient Accumulation | 1 |
| Max Length | 512 |
| Precision | FP16 |

## Usage

### With Transformers + PEFT

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "ylliprifti/hackathon-2025")
tokenizer = AutoTokenizer.from_pretrained("ylliprifti/hackathon-2025")

# Generate
prompt = "How do I use FLOWROLL to get a trailing 3-month total?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Merge Adapter (Optional)

```python
# Merge LoRA weights into base model for faster inference
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged-model")
```

## Training Data

This model was fine-tuned on **1,028 prompt-completion pairs** covering GridScript™ DSL usage:

- **Functions:** `FLOWROLL()` (rolling aggregations) and `DIMMATCH()` (dimensional alignment)
- **Question Types:** How-to guides, troubleshooting, syntax help, conceptual explanations
- **Tone Variations:** Casual, formal, technical, frustrated user, curious learner, problem-focused
- **Format:** Universal prompt-completion format (not chat templates)

### Data Composition

- Original training set: 429 examples
- Conceptual Q&A: 99 examples
- Augmented variations: 500 examples (10 batches with different tones)
- **Total:** 1,028 training examples

## What This Model Does

This model specializes in:
- ✅ Explaining GridScript™ `FLOWROLL()` and `DIMMATCH()` functions
- ✅ Troubleshooting common errors (blanks, dimension mismatches, period ordering)
- ✅ Providing correct syntax examples with proper parameters
- ✅ Understanding context from various question styles (casual to formal)
- ❌ **Not** a general-purpose model - trained exclusively on GridScript™ DSL

## Limitations

- **Domain-Specific:** Only trained on GridScript™ `FLOWROLL` and `DIMMATCH` functions
- **No Other Functions:** Does not know about other GridScript™ functions (SUM, IF, etc.)
- **Inherits Base Model Limitations:** Subject to Llama 3.2 3B's general limitations
- **Not Production-Ready:** Intended for hackathon/demo purposes without extensive evaluation
- **Fictional DSL:** GridScript™ is a fictional language created for this training project

## Example Queries

The model can answer questions like:

- "How do I use FLOWROLL to get a trailing 3-month total?"
- "Why does DIMMATCH fail when aligning revenue to customer list?"
- "What happens if I set PeriodCount to 1?"
- "FLOWROLL gives blanks for the first 5 periods. Why?"
- "Can I use DIMMATCH on time dimensions?"

## Training Details

- **Hardware:** NVIDIA RTX Quadro 8000 (48GB)
- **Training Time:** ~5 epochs
- **Optimization:** Pre-tokenized dataset for faster training
- **Loss Masking:** Only completion tokens used for loss (prompts masked with -100)
- **EOS Handling:** Model learns to generate proper end-of-sequence tokens

## License

This model is released under the MIT License. The base model (Llama 3.2) is subject to Meta's license terms.

---

*Fine-tuned using MLOps pipeline with LoRA, PEFT, and custom tokenization for DSL training*
