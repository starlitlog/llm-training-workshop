# Hardware Recommendations

Complete guide to choosing the right hardware for LLM training at different budget levels.

## 🎯 Quick Recommendations

| Budget | Configuration | Training Capability | Monthly Cost |
|--------|---------------|-------------------|--------------|
| **Budget** | RTX 3080 (12GB) + 32GB RAM | Llama-3B, LoRA training | $0 (own hardware) |
| **Prosumer** | RTX 4090 (24GB) + 64GB RAM | Llama-8B, full LoRA | $0 (own hardware) |
| **Cloud** | AWS p3.2xlarge (V100 16GB) | Llama-8B, flexible | $2,200/month |
| **Enterprise** | 8x H100 (640GB total) | Any model, any technique | $50,000/month |

## 💻 Desktop Builds

### **Budget Build (~$2,500)**
```
GPU:    RTX 3080 12GB       ($500-700)
CPU:    AMD Ryzen 7 7700X   ($300)
RAM:    32GB DDR5-5600      ($120)  
SSD:    2TB NVMe Gen4       ($150)
PSU:    850W 80+ Gold       ($120)
Mobo:   B650 chipset        ($150)
Case:   Mid-tower           ($80)
```

**Training Capability:**
- ✅ Llama-3B models comfortably
- ✅ Llama-7B with optimizations (batch_size=1)
- ⚠️ Llama-8B possible but slow
- ❌ Models >13B (insufficient VRAM)

### **Prosumer Build (~$4,500)**
```
GPU:    RTX 4090 24GB       ($1,600)
CPU:    AMD Ryzen 9 7950X   ($550)
RAM:    64GB DDR5-5600      ($220)
SSD:    4TB NVMe Gen4       ($300)
PSU:    1200W 80+ Platinum  ($200)
Mobo:   X670E chipset       ($300)
Case:   Full-tower          ($150)
```

**Training Capability:**
- ✅ Llama-8B models smoothly
- ✅ Llama-13B with careful memory management
- ✅ Multiple experiments in parallel
- ⚠️ Llama-30B+ (requires model sharding)

### **Enthusiast Build (~$8,000)**
```
GPU:    2x RTX 4090 24GB    ($3,200)
CPU:    AMD Threadripper    ($1,000)
RAM:    128GB DDR5-5600     ($400)
SSD:    8TB NVMe Gen4       ($600)
PSU:    1600W 80+ Titanium  ($400)
Mobo:   TRX50 chipset       ($600)
Case:   Workstation         ($300)
```

**Training Capability:**
- ✅ Any open-source model (up to 70B)
- ✅ Multi-GPU training and inference
- ✅ Production model serving
- ✅ Research and experimentation

## ☁️ Cloud Options

### **AWS EC2 GPU Instances**

| Instance | GPU | VRAM | RAM | CPU | Price/Hour | Best For |
|----------|-----|------|-----|-----|------------|----------|
| **g4dn.xlarge** | T4 | 16GB | 16GB | 4 vCPU | $0.526 | Testing, small models |
| **p3.2xlarge** | V100 | 16GB | 61GB | 8 vCPU | $3.06 | Llama-7B training |
| **p3.8xlarge** | 4x V100 | 64GB | 244GB | 32 vCPU | $12.24 | Multi-GPU training |
| **p4d.24xlarge** | 8x A100 | 320GB | 1.1TB | 96 vCPU | $32.77 | Enterprise training |

### **Google Cloud Platform**

| Instance | GPU | VRAM | RAM | CPU | Price/Hour | Best For |
|----------|-----|------|-----|-----|------------|----------|
| **n1-standard-4 + T4** | T4 | 16GB | 15GB | 4 vCPU | $0.95 | Budget training |
| **n1-standard-8 + V100** | V100 | 16GB | 30GB | 8 vCPU | $2.48 | Standard training |
| **a2-highgpu-1g** | A100 | 40GB | 85GB | 12 vCPU | $3.673 | Advanced training |
| **a2-highgpu-8g** | 8x A100 | 320GB | 680GB | 96 vCPU | $29.39 | Large-scale training |

### **Azure Machine Learning**

| Instance | GPU | VRAM | RAM | CPU | Price/Hour | Best For |
|----------|-----|------|-----|-----|------------|----------|
| **Standard_NC6s_v3** | V100 | 16GB | 112GB | 6 vCPU | $3.06 | Standard training |
| **Standard_NC24s_v3** | 4x V100 | 64GB | 448GB | 24 vCPU | $12.24 | Multi-GPU training |
| **Standard_ND96asr_v4** | 8x A100 | 320GB | 900GB | 96 vCPU | $27.20 | Enterprise training |

### **Specialized Platforms**

| Platform | GPU Options | Pricing | Pros | Cons |
|----------|-------------|---------|------|------|
| **Paperspace** | RTX 4000-6000, A4000-A6000 | $0.51-2.30/hr | Easy setup, Jupyter | Limited availability |
| **RunPod** | RTX 3090, 4090, A100 | $0.34-2.89/hr | Competitive pricing | Community-based |
| **Lambda Labs** | RTX 6000, A100, H100 | $1.50-8.00/hr | ML-optimized | Waitlists common |
| **CoreWeave** | RTX 3080-4090, A100 | $0.57-4.25/hr | Gaming GPU focus | Technical setup |

## 📊 Performance Comparisons

### **Training Speed (Llama-7B, 1000 steps)**

| GPU | VRAM | Batch Size | Time | Cost | Notes |
|-----|------|------------|------|------|-------|
| **RTX 3060** | 8GB | 1 | 45 min | $0 | gradient_checkpointing required |
| **RTX 3080** | 12GB | 2 | 28 min | $0 | Comfortable training |
| **RTX 4090** | 24GB | 4 | 15 min | $0 | Fast iteration |
| **V100** | 16GB | 2 | 22 min | $1.12 | Cloud option |
| **A100** | 40GB | 8 | 12 min | $0.73 | Professional |

### **Memory Requirements by Model**

| Model | Parameters | Minimum VRAM | Recommended VRAM | Notes |
|-------|------------|---------------|------------------|-------|
| **Llama-3.1-3B** | 3B | 6GB | 12GB | Perfect for learning |
| **Llama-3.1-8B** | 8B | 12GB | 24GB | Most versatile |
| **CodeLlama-13B** | 13B | 20GB | 40GB | Specialized models |
| **Llama-3.1-30B** | 30B | 48GB | 80GB | Multi-GPU required |
| **Llama-3.1-70B** | 70B | 80GB | 160GB | Enterprise setups |

### **Cost Analysis: Training a Custom Model**

**Scenario**: Fine-tune Llama-8B on 50K examples, 5 epochs

| Option | Setup | Training Time | Total Cost | Cost/Model |
|--------|-------|---------------|------------|------------|
| **RTX 4090 (owned)** | $1,600 upfront | 4 hours | $1,600 | $0.32 |
| **AWS p3.2xlarge** | $0 setup | 3 hours | $9.18 | $9.18 |
| **Google Colab Pro** | $10/month | 8 hours | $10 | $10 |
| **Paperspace A4000** | $0 setup | 5 hours | $11.50 | $11.50 |

*After 500 models, owned RTX 4090 becomes most economical*

## 🛠️ Optimization by Hardware

### **RTX 3060/3070 (8-12GB VRAM)**

```yaml
# Optimized config for budget GPUs
model_name: meta-llama/Llama-3.1-3B
batch_size: 1
gradient_accumulation_steps: 8
max_length: 256
gradient_checkpointing: true
bf16: false
fp16: true
lora_r: 16
lora_target_modules: [q_proj, v_proj]
```

**Expected Performance:**
- Training speed: ~30 tokens/second
- Memory usage: ~8GB VRAM
- Training time: 2-4 hours per epoch

### **RTX 3080/4080 (12-16GB VRAM)**

```yaml
# Balanced config for prosumer GPUs
model_name: meta-llama/Llama-3.1-8B
batch_size: 2
gradient_accumulation_steps: 4
max_length: 512
gradient_checkpointing: false
bf16: true  # RTX 4080 only
fp16: true  # RTX 3080
lora_r: 32
```

**Expected Performance:**
- Training speed: ~50 tokens/second
- Memory usage: ~14GB VRAM
- Training time: 1-2 hours per epoch

### **RTX 4090 (24GB VRAM)**

```yaml
# High-performance config
model_name: meta-llama/Llama-3.1-8B
batch_size: 4
gradient_accumulation_steps: 2
max_length: 512
gradient_checkpointing: false
bf16: true
lora_r: 64
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

**Expected Performance:**
- Training speed: ~80 tokens/second
- Memory usage: ~20GB VRAM
- Training time: 30-60 minutes per epoch

### **Multi-GPU Setups (2x RTX 4090)**

```yaml
# Multi-GPU distributed config
model_name: meta-llama/Llama-3.1-8B
batch_size: 8  # Per GPU
gradient_accumulation_steps: 1
max_length: 1024
bf16: true
```

```bash
# Launch with torchrun
torchrun --nproc_per_node=2 -m src.main train --config-name train_multi_gpu
```

## 💡 Buying Recommendations

### **For Beginners ($1,000-2,000 budget):**
1. **RTX 3080 12GB** - Best value for learning
2. Focus remaining budget on RAM (32GB minimum)
3. Fast NVMe SSD (1TB+) for datasets

### **For Serious Users ($3,000-5,000 budget):**
1. **RTX 4090 24GB** - Handles most models comfortably
2. High-end CPU for data preprocessing
3. 64GB RAM for large datasets
4. Multiple fast SSDs

### **For Professionals ($10,000+ budget):**
1. **2x RTX 4090** or **4x RTX 3090** for distributed training
2. Threadripper CPU with many cores
3. 128GB+ RAM
4. NVMe RAID arrays
5. Proper cooling and power infrastructure

## ⚡ Power & Cooling Considerations

### **Power Requirements**

| GPU | TDP | Recommended PSU | Peak Power |
|-----|-----|-----------------|------------|
| RTX 3060 | 170W | 600W | ~550W |
| RTX 3080 | 320W | 750W | ~700W |
| RTX 4090 | 450W | 1000W | ~950W |
| 2x RTX 4090 | 900W | 1500W | ~1400W |

### **Cooling Solutions**

**Air Cooling (Budget):**
- Large case with good airflow
- Multiple intake/exhaust fans
- Aftermarket GPU coolers if needed

**AIO Liquid Cooling (Recommended):**
- 240-360mm radiators for CPU
- Hybrid GPU coolers for extreme overclocking
- Better sustained performance

**Custom Loops (Enthusiast):**
- CPU + GPU in same loop
- Excellent performance and acoustics
- High maintenance, expensive

## 🔧 Setup Optimization Tips

### **BIOS Settings**
- Enable Resizable BAR
- Set PCIe to Gen4 (if supported)
- Enable XMP/DOCP for RAM
- Disable C-states for consistent performance

### **Software Optimization**
```bash
# NVIDIA driver optimizations
sudo nvidia-smi -pm 1  # Persistence mode
sudo nvidia-smi -ac 1215,210  # Memory/core clocks (RTX 4090)

# CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Huge pages for better memory performance
echo 2048 | sudo tee /proc/sys/vm/nr_hugepages
```

### **Monitoring**
```bash
# GPU monitoring
nvidia-smi -l 1

# System monitoring  
htop

# Power monitoring
sudo powerstat 1

# Temperature monitoring
sensors
```

## 📈 Future-Proofing Recommendations

### **Next-Generation GPUs (2024-2025)**
- **RTX 5090** - Expected 32GB VRAM, ~2x RTX 4090 performance
- **AMD RDNA 4** - Competitive pricing, unified memory
- **Intel Battlemage** - Budget option with good compute

### **Technology Trends**
- **More VRAM per dollar** - 24GB becoming standard
- **Better efficiency** - Lower power for same performance
- **Unified memory** - CPU/GPU sharing same memory pool
- **AI-specific hardware** - NPUs, TPUs becoming mainstream

---

## 🎯 Final Recommendations

**Just getting started?** RTX 3080 12GB + 32GB RAM

**Serious about LLM training?** RTX 4090 24GB + 64GB RAM

**Running a business on this?** 2x RTX 4090 or cloud with A100s

**Research institution?** 8x H100 cluster and unlimited budget 😄

Remember: The best GPU is the one you can afford and will actually use. Start smaller and upgrade as your skills and needs grow! 🚀