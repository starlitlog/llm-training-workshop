# Environment Setup Guide

Complete setup instructions for both slides and hands-on training components.

## 🖥️ System Requirements

### **Minimum Requirements**
- **CPU**: 8-core modern processor (Intel i7/AMD Ryzen 7)
- **RAM**: 32GB (16GB absolute minimum)
- **GPU**: RTX 3060 (8GB VRAM) for small models
- **Storage**: 100GB free space (models + datasets)

### **Recommended Setup**
- **CPU**: 16-core processor (Intel i9/AMD Ryzen 9)  
- **RAM**: 64GB for comfortable training
- **GPU**: RTX 4090 (24GB VRAM) or better
- **Storage**: 500GB NVMe SSD

### **Cloud Alternatives**
- **AWS**: p3.2xlarge (V100 16GB) - $3.06/hour
- **GCP**: n1-standard-8 + T4 (16GB) - $0.95/hour  
- **Google Colab Pro**: T4/P100 - $10/month (limited sessions)

## 📊 Slides Setup (5 minutes)

### Prerequisites
- Node.js 16+ ([Download](https://nodejs.org/))
- Git

### Installation
```bash
cd slides
npm install
```

### Run Slides
```bash
npm run dev
# Opens browser at http://localhost:5173
```

### Build for Production
```bash
npm run build
# Static files in dist/ folder
```

## 🔧 Hands-On Training Setup (15-30 minutes)

### Prerequisites

#### **CUDA Setup** (GPU required)
```bash
# Check CUDA version
nvidia-smi

# Install CUDA 11.8 if needed
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

#### **Python Environment**
```bash
# Python 3.10+ required
python3 --version

# Using pyenv (recommended)
pyenv install 3.10.8
pyenv global 3.10.8
```

### Installation

#### **Automated Setup**
```bash
cd hands-on
make venv && source .venv/bin/activate
make install
```

#### **Manual Setup**
```bash
cd hands-on

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

```bash
# Copy template and configure
cp .env.template .env

# Edit .env with your details
nano .env
```

**Required for HuggingFace Publishing**:
```bash
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"  # Get from https://huggingface.co/settings/tokens
HF_REPO_ID="your-username/your-model-name"
```

**Optional Configurations**:
```bash
CUDA_VISIBLE_DEVICES="0"                    # GPU selection
HF_HOME="~/.cache/huggingface"             # Cache directory  
WANDB_API_KEY="your-wandb-key"             # Experiment tracking
```

### Verify Installation

```bash
# Test basic functionality
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"

# Test model loading (downloads ~13GB)
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
print('✅ Model tokenizer loaded successfully')
"
```

## 🔗 Remote GPU Setup (Optional)

For training on remote GPU servers:

### **1. Configure Remote Access**
```bash
# Edit Makefile variables
REMOTE_USER := your_username
REMOTE_HOST := 192.168.1.100
REMOTE_DIR := ~/llm-training-workshop/hands-on
```

### **2. Setup SSH Keys**
```bash
# Generate SSH key (if needed)
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Copy to remote server
ssh-copy-id your_username@192.168.1.100
```

### **3. Initialize Remote Environment**
```bash
make push        # Sync code
make ssh-init    # Install dependencies
```

### **4. Run Remote Training**
```bash
# Background training (survives disconnect)
make ssh-train-bg TRAIN_CONFIG=train_llama3b

# Reconnect to session
ssh your_username@192.168.1.100 -t 'tmux attach -t train'
```

## 📦 Docker Setup (Alternative)

For containerized environments:

```bash
# Build container
docker build -t llm-training .

# Run with GPU support
docker run --gpus all -v $(pwd):/workspace llm-training
```

## 🧪 Test Installation

### **Quick Smoke Test**
```bash
cd hands-on

# Test data loading
python -c "from src.data.loader import load_dataset; print('✅ Data loading works')"

# Test configuration
python -m src.main list-configs

# Test tiny training run (5 minutes)
make train TRAIN_CONFIG=train_llama3b
```

### **Full Integration Test**
```bash
# Complete workflow test (~30 minutes)
make train TRAIN_CONFIG=train_llama3b
make eval
make eval-baseline
```

## 🐛 Common Issues

### **CUDA Out of Memory**
```bash
# Reduce batch size in config
batch_size: 1
gradient_accumulation_steps: 8

# Enable gradient checkpointing
gradient_checkpointing: true

# Use smaller model
make train TRAIN_CONFIG=train_llama3b
```

### **Model Download Fails**
```bash
# Set cache directory
export HF_HOME="/path/to/large/disk"

# Manual download
huggingface-cli download meta-llama/Llama-3.1-8B
```

### **Import Errors**
```bash
# Reinstall packages
pip install -r requirements.txt --force-reinstall

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### **Permission Errors**
```bash
# Fix permissions
chmod +x scripts/*.py
sudo chown -R $USER:$USER .venv/
```

## 📈 Performance Optimization

### **For RTX 3080 (12GB)**
```yaml
# Use train_llama3b.yaml
model_name: meta-llama/Llama-3.1-3B
batch_size: 2
gradient_accumulation_steps: 4
max_length: 256
gradient_checkpointing: true
```

### **For RTX 4090 (24GB)**  
```yaml
# Use train.yaml (default)
model_name: meta-llama/Llama-3.1-8B
batch_size: 4
gradient_accumulation_steps: 2
max_length: 512
gradient_checkpointing: false
```

### **For Cloud/Enterprise (80GB+)**
```yaml
# Custom config
model_name: meta-llama/Llama-3.1-8B
batch_size: 16
gradient_accumulation_steps: 1
max_length: 1024
```

## 🆘 Getting Help

1. **Check logs**: `tail -f outputs/runs/latest/logs/train.log`
2. **GPU monitoring**: `nvidia-smi -l 1`
3. **Memory usage**: `htop` or `free -h`
4. **Disk space**: `df -h`
5. **GitHub Issues**: Report bugs and ask questions

---

✅ **Setup complete!** You're ready to master LLM training.