# Troubleshooting Guide

Common issues and solutions for the LLM Training Workshop.

## 🔥 GPU & Memory Issues

### **CUDA Out of Memory (OOM)**

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions (in order of preference):**

1. **Reduce batch size**:
```yaml
# In your config file
batch_size: 1  # Start with 1, increase gradually
gradient_accumulation_steps: 8  # Keep effective batch size
```

2. **Enable gradient checkpointing**:
```yaml
gradient_checkpointing: true
```

3. **Reduce sequence length**:
```yaml
max_length: 256  # Instead of 512
```

4. **Use smaller model**:
```bash
make train TRAIN_CONFIG=train_llama3b  # 3B instead of 8B
```

5. **Reduce LoRA parameters**:
```yaml
lora_r: 16      # Instead of 64
lora_alpha: 32  # Instead of 128
lora_target_modules:  # Minimal set
  - q_proj
  - v_proj
```

### **CUDA Device Assertion Failed**

**Symptoms:**
```
RuntimeError: CUDA error: device-side assert triggered
```

**Solutions:**
```bash
# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1

# Check for NaN values in data
python -c "
import torch
from src.data.loader import load_dataset
data = load_dataset('./data')
print('Data contains NaN:', torch.isnan(data).any())
"

# Restart training from scratch
rm -rf outputs/runs/latest
make train
```

### **GPU Not Detected**

**Symptoms:**
```
CUDA Available: False
```

**Solutions:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Check GPU visibility
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print('GPU Count:', torch.cuda.device_count())"
```

## 📊 Training Issues

### **Loss Not Decreasing**

**Symptoms:**
- Loss stays flat or oscillates
- No improvement over baseline

**Diagnostics:**
```bash
# Check learning rate
tensorboard --logdir outputs/runs/latest/logs

# Verify data loading
python -c "
from src.data.loader import load_dataset
data = load_dataset('./data')
print('Dataset size:', len(data))
print('Sample:', data[0])
"

# Check gradient norms
# (Look in TensorBoard for grad_norm values)
```

**Solutions:**

1. **Adjust learning rate**:
```yaml
lr: 1e-4  # Try 10x smaller
lr: 1e-3  # Try 10x larger
```

2. **Check data format**:
```json
// Correct format
{"prompt": "What is FLOWROLL?", "completion": "FLOWROLL is a function..."}

// Incorrect format  
{"input": "What is FLOWROLL?", "output": "FLOWROLL is a function..."}
```

3. **Increase training epochs**:
```yaml
epochs: 10  # Instead of 5
```

4. **Verify tokenization**:
```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
text = 'What is FLOWROLL?'
tokens = tokenizer.encode(text)
print('Tokens:', len(tokens))
print('Decoded:', tokenizer.decode(tokens))
"
```

### **Training Extremely Slow**

**Symptoms:**
- < 1 step per minute
- Low GPU utilization

**Solutions:**

1. **Optimize data loading**:
```yaml
dataloader_num_workers: 8      # Use all CPU cores
dataloader_prefetch_factor: 4   # Pre-fetch batches
```

2. **Use pre-tokenized data**:
```bash
make tokenize TRAIN_CONFIG=train_llama3b
# Edit config: use_tokenized: true
```

3. **Check I/O bottlenecks**:
```bash
# Move data to faster storage
sudo mount -t tmpfs -o size=20G tmpfs /tmp/data
cp -r data/* /tmp/data/
# Update dataset_path in config
```

4. **Enable mixed precision**:
```yaml
bf16: true   # For RTX 4090
fp16: true   # For RTX 3080
```

### **Training Unstable/Diverging**

**Symptoms:**
- Loss suddenly spikes
- NaN values appear
- Model outputs gibberish

**Solutions:**

1. **Reduce learning rate**:
```yaml
lr: 1e-4  # Conservative value
warmup_ratio: 0.1  # Longer warmup
```

2. **Add gradient clipping**:
```yaml
max_grad_norm: 1.0  # Add to config
```

3. **Check data quality**:
```bash
python scripts/validate_data.py
```

## 📁 Data & Configuration Issues

### **Data Loading Errors**

**Symptoms:**
```
FileNotFoundError: No JSONL files found
```

**Solutions:**
```bash
# Check data directory
ls -la data/
ls -la data/eval/

# Verify JSONL format
head -n 1 data/batch-1.jsonl | python -m json.tool

# Convert data if needed
make convert

# Check file permissions
chmod 644 data/*.jsonl
```

### **Configuration Not Found**

**Symptoms:**
```
MissingConfigException: Cannot find primary config 'train_custom'
```

**Solutions:**
```bash
# List available configs
python -m src.main list-configs

# Check config syntax
python -c "import yaml; yaml.safe_load(open('configs/train.yaml'))"

# Use correct config name
make train TRAIN_CONFIG=train  # Not train.yaml
```

### **Import Errors**

**Symptoms:**
```
ModuleNotFoundError: No module named 'src'
```

**Solutions:**
```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use module syntax
python -m src.main train

# Check virtual environment
which python
pip list | grep transformers
```

## 🌐 Remote Training Issues

### **SSH Connection Failed**

**Symptoms:**
```
ssh: connect to host 192.168.1.100 port 22: Connection refused
```

**Solutions:**
```bash
# Test SSH connection
ssh -v your_username@192.168.1.100

# Check SSH service on remote
sudo systemctl status ssh

# Check firewall
sudo ufw status

# Use SSH keys instead of password
ssh-copy-id your_username@192.168.1.100
```

### **Remote Sync Issues**

**Symptoms:**
```
rsync: command not found
```

**Solutions:**
```bash
# Install rsync locally and remotely
# Ubuntu/Debian:
sudo apt install rsync

# macOS:
brew install rsync

# Manual sync alternative
scp -r . your_username@192.168.1.100:~/llm-training-workshop/
```

### **Tmux Session Issues**

**Symptoms:**
```
tmux: command not found
session not found: train
```

**Solutions:**
```bash
# Install tmux on remote server
sudo apt install tmux

# List sessions
ssh your_username@192.168.1.100 'tmux list-sessions'

# Kill stuck session
ssh your_username@192.168.1.100 'tmux kill-session -t train'

# Start new session manually
ssh your_username@192.168.1.100 'tmux new-session -d -s train'
```

## 📊 Evaluation Issues

### **Evaluation Metrics All Zero**

**Symptoms:**
- ROUGE, BLEU, exact_match all show 0.0
- Model seems to output nothing

**Solutions:**

1. **Check evaluation data format**:
```json
// Correct format for evaluation
{"prompt": "Question here?", "completion": "Expected answer"}
```

2. **Verify model outputs**:
```bash
python -c "
from src.eval.evaluator import evaluate_model
# Add debug prints to see actual vs expected outputs
"
```

3. **Check tokenizer configuration**:
```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('outputs/runs/latest/model')
print('Pad token:', tokenizer.pad_token)
print('EOS token:', tokenizer.eos_token)
"
```

### **Baseline Evaluation Fails**

**Symptoms:**
```
Model not found: meta-llama/Llama-3.1-8B
```

**Solutions:**
```bash
# Download base model first
huggingface-cli download meta-llama/Llama-3.1-8B

# Check HuggingFace authentication
huggingface-cli whoami

# Use different base model
# Edit eval_baseline.yaml:
model_name: meta-llama/Llama-3.1-3B  # Publicly available
```

## 💾 Storage & Performance

### **Disk Space Issues**

**Symptoms:**
```
No space left on device
```

**Solutions:**
```bash
# Check disk usage
df -h
du -sh outputs/
du -sh ~/.cache/huggingface/

# Clean old runs
rm -rf outputs/runs/2024-*  # Keep only latest

# Clean model cache
rm -rf ~/.cache/huggingface/transformers/
rm -rf ~/.cache/huggingface/hub/

# Move cache to larger disk
export HF_HOME="/path/to/large/disk/.cache"
```

### **Slow File I/O**

**Symptoms:**
- Data loading takes forever
- High iowait in `top`

**Solutions:**
```bash
# Use faster storage
sudo mount -t tmpfs -o size=50G tmpfs /tmp/training
cp -r data /tmp/training/
# Update config: dataset_path: /tmp/training/data

# Optimize for SSDs
echo deadline | sudo tee /sys/block/nvme0n1/queue/scheduler

# Pre-tokenize datasets
make tokenize TRAIN_CONFIG=train_llama3b
```

## 🆘 Emergency Procedures

### **Training Hung/Frozen**

```bash
# Check if process is still alive
ps aux | grep python

# Check GPU activity
nvidia-smi

# Kill training process
pkill -f "python -m src.main train"

# Clean up CUDA processes
sudo fuser -v /dev/nvidia*
sudo kill -9 <process_ids>

# Restart training
make train TRAIN_CONFIG=train_llama3b
```

### **Corrupted Model Files**

```bash
# Check file integrity
ls -la outputs/runs/latest/model/
file outputs/runs/latest/model/*.safetensors

# Restore from backup
cp -r artifacts/models/latest outputs/runs/latest/model

# Start fresh training
rm -rf outputs/runs/latest
make train TRAIN_CONFIG=train_llama3b
```

### **Complete Environment Reset**

```bash
# Nuclear option - start completely fresh
rm -rf .venv
rm -rf outputs
rm -rf ~/.cache/huggingface

# Reinstall everything
make venv && source .venv/bin/activate
make install
make train TRAIN_CONFIG=train_llama3b
```

## 📧 Getting More Help

If none of these solutions work:

1. **Enable debug logging**:
```bash
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
python -m src.main train --config-name train_llama3b 2>&1 | tee debug.log
```

2. **Gather system information**:
```bash
nvidia-smi > system_info.txt
python --version >> system_info.txt
pip list >> system_info.txt
echo "Disk space:" >> system_info.txt
df -h >> system_info.txt
```

3. **Create GitHub Issue** with:
   - Full error message
   - System information
   - Configuration used
   - Steps to reproduce

---

**Remember**: Most issues are memory-related. Start with smaller models and configurations, then scale up once everything works! 🚀