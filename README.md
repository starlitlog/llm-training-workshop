# LLM Training Workshop
> **Master practical LLM training with real production tools**

A complete learning path from theory to production: comprehensive slides + hands-on MLOps pipeline for fine-tuning Large Language Models.

## 🎯 Complete Learning Journey

| Component | Purpose | What You Get |
|-----------|---------|--------------|
| **[📊 Slides](./slides/)** | Theory & Understanding | Why custom training beats APIs, when to train, cost analysis |
| **[🔧 Hands-On](./hands-on/)** | Practical Implementation | Production MLOps pipeline with LoRA fine-tuning |
| **[📚 Examples](./examples/)** | Real-World Data | Multiple training scenarios and datasets |

## 🚀 Quick Start

### 1. **Understand the Theory** (15 minutes)
```bash
cd slides
npm install && npm run dev
# Navigate through slides to understand LLM training fundamentals
```

### 2. **Try Hands-On Training** (30 minutes)
```bash
cd hands-on
make venv && source .venv/bin/activate
make install

# Train a model on custom DSL (proves concept)
make train TRAIN_CONFIG=train_llama3b

# Evaluate against baseline
make eval && make eval-baseline
```

### 3. **Explore Real Examples**
```bash
cd examples
# Coming soon: Kaggle datasets and realistic scenarios
```

## 💡 Why This Workshop?

### **The Problem**: 
Most companies burn $100K+/month on API calls for tasks that could cost $50/month with custom training.

### **The Solution**:
Learn **when and how** to train custom models that beat GPT-4 on your specific domain.

### **Perfect Example**: 
The included **FLOWROLL/DIMMATCH DSL training** demonstrates teaching models completely novel concepts that don't exist anywhere else - exactly when custom training is essential.

## 🎓 What You'll Learn

### **From Slides** (Theory):
- **When to train vs. use APIs** - Cost/benefit analysis
- **LLM training fundamentals** - From perceptrons to Transformers  
- **Critical hyperparameters** - Learning rate, batch size, LoRA configuration
- **Memory optimization** - Training on consumer GPUs
- **Profiling & monitoring** - Production debugging techniques

### **From Hands-On** (Practice):
- **Production MLOps pipeline** - Real tools, not toys
- **LoRA fine-tuning** - Memory-efficient training
- **Experiment tracking** - TensorBoard, metrics, versioning
- **Model evaluation** - ROUGE, BLEU, exact match scoring
- **Deployment workflow** - HuggingFace Hub publishing
- **Remote GPU training** - Scale to cloud instances

## 🎯 Learning Path

```
Theory → Practice → Real Applications

📊 Slides        🔧 Hands-On         📚 Examples
├─ Why train?    ├─ Setup pipeline   ├─ Technical docs
├─ When worth?   ├─ Train DSL model  ├─ Financial data  
├─ How works?    ├─ Evaluate model   ├─ Customer support
└─ Cost trade    └─ Deploy model     └─ Legal analysis
```

## 📈 Success Metrics

By the end, you'll be able to:

- ✅ **Identify when custom training beats APIs** (cost/capability analysis)
- ✅ **Train models on consumer GPUs** (RTX 3080 → RTX 4090)
- ✅ **Implement production MLOps** (experiment tracking, evaluation, deployment)
- ✅ **Optimize for your budget** (memory tricks, GPU selection, cloud costs)
- ✅ **Deploy custom models confidently** (evaluation metrics, baseline comparison)

## 🔧 System Requirements

### **For Slides** (any machine):
- Node.js 16+
- Web browser

### **For Hands-On** (GPU recommended):
- **Minimum**: RTX 3080 (12GB VRAM) + 32GB RAM
- **Recommended**: RTX 4090 (24GB VRAM) + 64GB RAM  
- **Cloud**: AWS p3.2xlarge, GCP n1-standard-8 + T4
- Python 3.10+, CUDA 11.8+

### **Tested Configurations**:
- ✅ RTX 4090 (24GB) - Full Llama-8B training
- ✅ RTX 3080 (12GB) - Llama-3B with optimizations
- ✅ Google Colab Pro - Llama-3B (limited session time)

## 📖 Project Structure

```
llm-training-workshop/
├── README.md                    # This comprehensive overview
├── LICENSE                      # MIT License for the entire project
├── slides/                      # 📊 Interactive React presentation
│   ├── src/App.tsx             # 26 comprehensive slides covering theory
│   ├── package.json            # Vite + React + Tailwind stack
│   └── README.md               # Detailed slides setup & customization
├── hands-on/                    # 🔧 Production MLOps pipeline  
│   ├── src/                    # Python training & evaluation code
│   ├── configs/                # Hydra configurations (multiple GPU setups)
│   ├── data/                   # FLOWROLL/DIMMATCH DSL training examples
│   ├── Makefile                # Easy commands for all operations
│   ├── requirements.txt        # Python dependencies
│   └── README.md               # Complete hands-on setup & usage guide
├── docs/                        # 📚 Comprehensive shared documentation
│   ├── SETUP.md                # Complete environment setup guide
│   ├── TROUBLESHOOTING.md      # Common issues & solutions
│   └── HARDWARE.md             # GPU recommendations & performance
└── examples/                    # 🚀 Additional training scenarios
    ├── README.md               # Overview of available examples
    ├── technical-docs/         # API documentation training (planned)
    ├── financial-analysis/     # Domain-specific language (planned)
    └── customer-support/       # Company-specific training (planned)
```

### **Component Details**

#### **📊 [Slides](./slides/)** 
Interactive React presentation covering LLM training from theory to production:
- **26 comprehensive slides** covering the complete journey
- **Cost analysis tools** showing $100K API vs $50 training scenarios  
- **LLM evolution history** from perceptrons to Transformers
- **Production deployment** strategies and monitoring
- **Live demo integration** with hands-on project
- **Customizable content** for different audiences
- **Mobile responsive** design for any device

**Quick Start**: `cd slides && npm install && npm run dev`

#### **🔧 [Hands-On](./hands-on/)**
Production-ready MLOps pipeline for practical LLM training:
- **LoRA fine-tuning** with memory-efficient PEFT library
- **Multiple GPU configurations** (RTX 3080, 4090, A100, etc.)
- **Custom DSL training example** (FLOWROLL/DIMMATCH)
- **Comprehensive evaluation** (ROUGE, BLEU, exact match)
- **Remote GPU training** with SSH automation
- **HuggingFace integration** for model publishing
- **Experiment tracking** with TensorBoard
- **Professional MLOps** workflows

**Quick Start**: `cd hands-on && make venv && source .venv/bin/activate && make install && make train`

#### **📚 [Documentation](./docs/)**
Complete setup and troubleshooting guides:
- **[SETUP.md](./docs/SETUP.md)** - Environment setup for both components
- **[TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[HARDWARE.md](./docs/HARDWARE.md)** - GPU recommendations and performance comparisons

#### **🚀 [Examples](./examples/)**
Additional training scenarios for real-world applications:
- **Current**: FLOWROLL/DIMMATCH custom DSL (perfect novel concept example)
- **Planned**: Technical documentation, financial analysis, customer support, legal documents

## 🌟 Why This Combination Works

1. **Slides provide context** - Understand the "why" and "when"
2. **Hands-on provides skills** - Learn the "how" with real tools  
3. **Examples provide confidence** - Apply to realistic scenarios
4. **Everything stays in sync** - Theory matches practice

## 🚀 Getting Started

Choose your learning style:

### **📚 Academic Approach**:
1. Read through slides completely
2. Review hands-on documentation  
3. Start with smallest model configuration
4. Gradually increase complexity

### **🔧 Practical Approach**:
1. Jump into hands-on training immediately
2. Reference slides when you hit concepts
3. Experiment with different configurations
4. Try additional examples when confident

### **⚡ Quick Demo** (10 minutes):
```bash
# See the slides
cd slides && npm run dev

# Train a tiny model
cd hands-on && make train TRAIN_CONFIG=train_llama3b
```

## 💬 Community & Support

- **Issues**: Use GitHub Issues for bugs and questions
- **Discussions**: Share your training results and learnings  
- **Discord**: [Link coming soon] for real-time help
- **Contributions**: See [hands-on/CONTRIBUTING.md](./hands-on/CONTRIBUTING.md)

## 📄 License & Attribution

[MIT License](./LICENSE) - **Feel free to copy, use, modify, and distribute this workshop for any purpose.**

### **Creator & Attribution**

Created by **Dr. Ylli Prifti** ([ORCID: 0000-0002-9323-875X](https://orcid.org/0000-0002-9323-875X))

**When using this work, please provide credit:**
- **For academic/research use**: Cite the creator and provide a link to this repository
- **For commercial/workshop use**: Include attribution to "Dr. Ylli Prifti" in materials
- **For modifications**: Mention "Based on LLM Training Workshop by Dr. Ylli Prifti"

**Suggested citation format:**
```
Prifti, Y. (2025). LLM Training Workshop: Complete Learning Path from Theory to Production. 
GitHub repository. https://github.com/starlitlog/llm-training-workshop
```

### **Academic Publication**

This work has been developed into a comprehensive **academic paper** covering:
- **Integrated educational frameworks** for practical LLM training
- **Novel domain training methodologies** with the FLOWROLL/DIMMATCH DSL case study
- **Hardware-conscious education** approaches for consumer GPU accessibility
- **Production-ready MLOps pipelines** for educational implementation
- **Cost-effectiveness analysis** of custom training vs. commercial APIs

**📄 ArXiv link**: *Pending publication*

---

## 🎉 Ready to Master LLM Training?

**Start with the slides to understand the fundamentals, then dive into hands-on practice with real production tools.**

The custom DSL example will blow your mind - it shows exactly when and why you need custom training over APIs. 

*Stop paying API rent. Start owning your AI infrastructure.* 🚀

**This work represents a complete, production-ready learning system for mastering LLM training - from theory to practice.**