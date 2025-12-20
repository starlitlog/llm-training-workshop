# Training Examples & Datasets

Additional training scenarios and datasets to complement the hands-on FLOWROLL/DIMMATCH examples.

## 🎯 Available Examples

### **1. FLOWROLL/DIMMATCH DSL** (Included in hands-on/)
**Perfect example of novel concept training**

- **Purpose**: Demonstrate training on completely new concepts
- **Business Case**: Internal company DSL, proprietary tools  
- **Data Size**: 23 batches (~2,300 examples)
- **Training Time**: 30-60 minutes on RTX 4090
- **Key Learning**: When you NEED custom training vs APIs

```bash
cd ../hands-on
make train TRAIN_CONFIG=train_llama3b
```

## 🚀 Coming Soon: Realistic Kaggle Datasets

### **Technical Documentation Q&A**
**Training on API documentation and technical guides**

- **Dataset**: Stack Overflow Python + GitHub Issues  
- **Size**: ~100K examples
- **Business Case**: Internal company API documentation
- **Value**: Replace expensive technical writers

```
examples/technical-docs/
├── setup.py          # Download and prepare data
├── preprocess.py      # Clean and format for training  
├── config.yaml        # Optimized training config
└── README.md          # Setup instructions
```

### **Financial Domain Specialization**  
**Training on financial language and analysis**

- **Dataset**: SEC Filings + Financial News
- **Size**: ~50K examples
- **Business Case**: Investment analysis, risk assessment
- **Value**: Domain expertise that GPT-4 lacks

```
examples/financial-analysis/
├── download_sec.py    # SEC filing scraper
├── financial_config.yaml
└── evaluation/       # Finance-specific metrics
```

### **Customer Support Automation**
**Company-specific support ticket training**

- **Dataset**: E-commerce support tickets (anonymized)
- **Size**: ~75K examples  
- **Business Case**: Reduce support costs by 80%
- **Value**: Company-specific products and policies

```
examples/customer-support/
├── synthetic_tickets.py  # Generate realistic data
├── support_config.yaml
└── metrics/             # Support-specific evaluation
```

### **Legal Document Analysis**
**Specialized legal language and reasoning**

- **Dataset**: Legal case summaries + contract analysis
- **Size**: ~30K examples
- **Business Case**: Law firms, compliance departments  
- **Value**: Legal reasoning that requires domain training

```
examples/legal-analysis/
├── legal_corpus.py    # Legal document processing
├── legal_config.yaml  
└── evaluation/       # Legal-specific metrics
```

## 📊 Training Comparison Matrix

| Example | Data Size | Training Time | Business Value | API Alternative Cost |
|---------|-----------|---------------|----------------|---------------------|
| **FLOWROLL DSL** | 2.3K | 30 min | Impossible to outsource | N/A - doesn't exist |
| **Technical Docs** | 100K | 2 hours | $50K/year tech writer | $10K/month API calls |
| **Financial** | 50K | 1.5 hours | $200K/year analyst | $25K/month specialized API |
| **Support** | 75K | 1.8 hours | 80% ticket reduction | $15K/month support API |
| **Legal** | 30K | 1 hour | $150K/year paralegal | $20K/month legal AI |

## 🛠️ Setup Instructions

### **Prerequisites**
```bash
# Install additional dependencies
pip install kaggle datasets beautifulsoup4 requests

# Configure Kaggle API
# 1. Create account at kaggle.com
# 2. Go to Account → API → Create New Token
# 3. Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### **Download and Prepare Example Dataset**
```bash
# Choose an example
cd examples/technical-docs

# Download and prepare data
python setup.py

# Start training
cd ../../hands-on
make train TRAIN_CONFIG=../examples/technical-docs/config.yaml
```

### **Create Your Own Example**
```bash
# Copy template
cp -r examples/template examples/my-custom-domain

# Edit configuration
nano examples/my-custom-domain/config.yaml

# Add your training data
# Format: {"prompt": "Question", "completion": "Answer"}
```

## 📈 Performance Expectations

### **Training Speed by Example**

| Dataset | RTX 3080 (12GB) | RTX 4090 (24GB) | A100 (40GB) |
|---------|------------------|------------------|--------------|
| **FLOWROLL (2K)** | 25 min | 15 min | 12 min |
| **Technical (100K)** | 4 hours | 2.5 hours | 2 hours |
| **Financial (50K)** | 2.5 hours | 1.5 hours | 1.2 hours |
| **Support (75K)** | 3.5 hours | 2.2 hours | 1.8 hours |
| **Legal (30K)** | 1.8 hours | 1.1 hours | 55 min |

### **Memory Requirements**

| Dataset Size | Minimum VRAM | Recommended VRAM | Batch Size |
|--------------|---------------|-------------------|------------|
| **< 10K** | 8GB | 12GB | 2-4 |
| **10K-50K** | 12GB | 16GB | 2-3 |
| **50K-100K** | 16GB | 24GB | 1-2 |
| **100K+** | 20GB | 32GB | 1 |

## 🎓 Educational Value

### **Learning Progression**

1. **Start with FLOWROLL** - Understand custom domain training
2. **Try Technical Docs** - Scale up to realistic data sizes  
3. **Experiment with Financial** - Domain-specific language challenges
4. **Advanced: Legal/Support** - Complex reasoning and context

### **Key Concepts Demonstrated**

- **Data preprocessing** - Real-world data is messy
- **Domain adaptation** - Generic models → specialized experts
- **Evaluation metrics** - Domain-specific success measures
- **Cost optimization** - Training vs API economics
- **Production deployment** - From training to serving

## 🤝 Contributing New Examples

We welcome contributions of new training examples! See [CONTRIBUTING.md](../hands-on/CONTRIBUTING.md) for guidelines.

**Ideal contributions:**
- **Realistic business scenarios** - Problems companies actually face
- **Public datasets** - No proprietary or sensitive data
- **Clear value proposition** - When custom training beats APIs
- **Complete setup** - Download, preprocess, train, evaluate
- **Documentation** - Business case, technical details, troubleshooting

## 📚 Resources

- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **HuggingFace Datasets**: https://huggingface.co/datasets
- **Papers With Code**: https://paperswithcode.com/datasets
- **Google Dataset Search**: https://datasetsearch.research.google.com/

---

**Remember**: The goal isn't just to train models - it's to **demonstrate clear business value** where custom training beats expensive API calls! 🚀