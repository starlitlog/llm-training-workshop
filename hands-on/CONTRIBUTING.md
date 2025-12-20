# Contributing to Hands-On LLM Training

Thank you for your interest in contributing! This project aims to provide a practical, production-ready pipeline for LLM fine-tuning that serves as a companion to LLM training workshops.

## 🎯 Project Goals

- **Educational**: Help people learn practical LLM training
- **Production-Ready**: Real MLOps workflows, not just toys
- **Accessible**: Work on consumer GPUs, not just enterprise hardware
- **Custom Domain Focus**: Demonstrate training on novel/proprietary concepts

## 🚀 Quick Start for Contributors

1. **Fork and clone the repository**
```bash
git clone https://github.com/your-username/hands-on-llm-training.git
cd hands-on-llm-training
```

2. **Set up development environment**
```bash
make venv
source .venv/bin/activate
make install
```

3. **Run tests** (when available)
```bash
make test
```

## 📝 Types of Contributions We Welcome

### 🔧 **Code Improvements**
- Memory optimization for smaller GPUs
- Training speed optimizations
- Better error handling and logging
- Additional evaluation metrics

### 📊 **Configuration Examples**
- New GPU configurations (RTX 3060, RTX 4080, etc.)
- Different model size setups (3B, 13B, 70B)
- Cloud platform configs (AWS, GCP, Azure)

### 📖 **Documentation**
- Beginner tutorials
- Troubleshooting guides
- Cost optimization guides
- Hardware recommendations

### 🎓 **Educational Content**
- Additional training datasets for different domains
- Jupyter notebook tutorials
- Video walkthroughs
- Workshop materials

### 🐛 **Bug Reports**
- Memory issues on specific GPUs
- Training instabilities
- Configuration errors
- Environment setup problems

## 💡 Contribution Guidelines

### **Code Style**
- Use `black` for Python formatting: `make format`
- Run `make lint` before submitting
- Add type hints where appropriate
- Write docstrings for new functions

### **Configuration Files**
- Use clear, descriptive parameter names
- Add comments explaining non-obvious settings
- Include memory requirements in filename/comments
- Test on actual hardware when possible

### **Documentation**
- Write for beginners who are new to LLM training
- Include concrete examples with expected outputs
- Explain the "why" behind configurations
- Link to relevant sections in the LLM training slides

### **Training Data**
- Keep examples educational and non-controversial
- Ensure data demonstrates real learning challenges
- Include both simple and complex examples
- Consider licensing/copyright issues

## 🔄 Pull Request Process

1. **Create a descriptive branch name**
   - `feature/rtx-3080-config`
   - `fix/memory-leak-tokenizer`
   - `docs/troubleshooting-guide`

2. **Write clear commit messages**
   ```
   Add RTX 3080 training configuration
   
   - Optimized for 12GB VRAM
   - Tested with Llama-7B model
   - Includes memory monitoring
   ```

3. **Test your changes**
   - Run existing tests
   - Test on relevant hardware if possible
   - Include example outputs/logs

4. **Update documentation**
   - Add new configs to README
   - Update relevant guides
   - Include troubleshooting notes

5. **Submit pull request**
   - Reference related issues
   - Describe testing performed
   - Include performance/memory metrics

## 🎯 Priority Areas

We're especially looking for contributions in:

1. **Consumer GPU Support**
   - RTX 3060, 3070, 3080, 3090
   - RTX 4060, 4070, 4080, 4090
   - Memory optimization techniques

2. **Cloud Platform Integration**
   - AWS EC2 G/P instances
   - Google Cloud GPU instances
   - Azure ML compute

3. **Educational Materials**
   - Step-by-step tutorials
   - Common error solutions
   - Cost analysis tools

4. **Additional Model Support**
   - Code generation models
   - Different model families
   - Quantization examples

## 🐛 Reporting Issues

When reporting bugs, please include:

- **Hardware**: GPU model, VRAM, system specs
- **Software**: CUDA version, Python version, package versions
- **Configuration**: Which config file you used
- **Error logs**: Full error messages and stack traces
- **Steps to reproduce**: Exact commands run

Use this template:

```
## Hardware
- GPU: RTX 4090 (24GB VRAM)
- CPU: Intel i9-12900K
- RAM: 32GB

## Software
- CUDA: 12.1
- Python: 3.10.8
- PyTorch: 2.1.0

## Issue
Brief description of the problem...

## Steps to Reproduce
1. `make venv && source .venv/bin/activate`
2. `make train TRAIN_CONFIG=train_consumer_gpu`
3. Error occurs after ~10 minutes

## Error Log
```
[full error message here]
```
```

## 📚 Resources

- **LLM Training Slides**: [Link to companion slides]
- **HuggingFace PEFT Docs**: https://huggingface.co/docs/peft
- **PyTorch Documentation**: https://pytorch.org/docs
- **Hydra Configuration**: https://hydra.cc/docs

## 💬 Community

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discord**: [Workshop Discord server] for real-time chat

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make LLM training more accessible! 🚀