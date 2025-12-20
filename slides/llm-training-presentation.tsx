import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const slides = [
  {
    title: "Training LLMs from Base Models",
    subtitle: "Foundations & Architecture",
    presenterNotes: [
      "Welcome to training session on LLMs from base models",
      "Cover foundations, architecture, and practical implementation",
      "Focus on Transformers, LoRA fine-tuning, and MLOps pipeline",
      "This is a hands-on workshop with practical examples"
    ],
    content: (
      <div className="flex flex-col items-center justify-center h-full">
        <h1 className="text-4xl font-bold text-white mb-4">Training LLMs from Base Models</h1>
        <h2 className="text-xl text-blue-300 mb-8">Session 1: Foundations & Architecture</h2>
        <div className="flex gap-4 mt-8">
          <span className="px-4 py-2 bg-blue-900/50 rounded-lg text-blue-200">Transformers</span>
          <span className="px-4 py-2 bg-purple-900/50 rounded-lg text-purple-200">LoRA</span>
          <span className="px-4 py-2 bg-green-900/50 rounded-lg text-green-200">MLOps</span>
        </div>
      </div>
    )
  },
  {
    title: "Historical Evolution",
    subtitle: "1943-1990s",
    presenterNotes: [
      "Start with McCulloch-Pitts (1943) - first mathematical neural model",
      "Perceptron (1958) could only solve linearly separable problems",
      "XOR problem exposed limitations - killed first AI wave",
      "Symbolic AI dominated 60s-80s with expert systems",
      "Backpropagation (1986) revival enabled training deep networks",
      "Universal Approximation Theorem proved theoretical foundation"
    ],
    content: (
      <div className="space-y-3">
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h3 className="text-base font-semibold text-blue-400 mb-1">McCulloch-Pitts & Perceptron</h3>
          <p className="text-gray-300 text-sm">1943: First neural model • 1958: Perceptron (linear classifier)</p>
          <p className="text-gray-400 text-xs mt-1">XOR problem killed first AI wave (Minsky & Papert, 1969)</p>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-800/50 p-3 rounded-xl">
            <h4 className="text-purple-400 font-semibold text-sm">Symbolic AI</h4>
            <p className="text-gray-400 text-xs">Expert Systems: DENDRAL, MYCIN</p>
            <p className="text-gray-400 text-xs">IF-THEN rules, Fuzzy Logic</p>
          </div>
          <div className="bg-gray-800/50 p-3 rounded-xl">
            <h4 className="text-green-400 font-semibold text-sm">Revival (1986)</h4>
            <p className="text-gray-400 text-xs">Backpropagation: Hinton et al.</p>
            <p className="text-gray-400 text-xs">Universal Approximation (1989)</p>
          </div>
        </div>
        <div className="bg-yellow-900/30 border border-yellow-600/50 p-2 rounded-xl">
          <p className="text-yellow-200 text-xs">💡 Backprop: Chain rule gradients from output to input. Like git blame for neural network errors.</p>
        </div>
      </div>
    )
  },
  {
    title: "Deep Learning Explosion",
    subtitle: "2010-2015: Key Innovations",
    presenterNotes: [
      "ReLU activation solved vanishing gradient problem",
      "Dropout prevented overfitting by forcing redundancy",
      "Batch normalization enabled stable deep training",
      "Adam optimizer provided adaptive learning rates",
      "AlexNet (2012) proved CNNs could dominate computer vision"
    ],
    content: (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-blue-900/40 p-3 rounded-xl">
            <h4 className="text-blue-300 font-semibold text-sm">ReLU</h4>
            <code className="text-green-300 text-xs">f(x) = max(0, x)</code>
            <p className="text-gray-400 text-xs">No vanishing gradient</p>
          </div>
          <div className="bg-purple-900/40 p-3 rounded-xl">
            <h4 className="text-purple-300 font-semibold text-sm">Dropout (2012)</h4>
            <p className="text-gray-400 text-xs">Random neuron zeroing</p>
            <p className="text-gray-400 text-xs">Forces redundancy</p>
          </div>
          <div className="bg-green-900/40 p-3 rounded-xl">
            <h4 className="text-green-300 font-semibold text-sm">BatchNorm (2015)</h4>
            <p className="text-gray-400 text-xs">Normalize layer inputs</p>
            <p className="text-gray-400 text-xs">Stable deep training</p>
          </div>
          <div className="bg-orange-900/40 p-3 rounded-xl">
            <h4 className="text-orange-300 font-semibold text-sm">Adam (2014)</h4>
            <p className="text-gray-400 text-xs">Adaptive learning rates</p>
            <p className="text-gray-400 text-xs">Per-parameter momentum</p>
          </div>
        </div>
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <p className="text-yellow-400 font-semibold text-sm">AlexNet (2012): 16.4% error vs 25.8% — CNNs dominate vision</p>
        </div>
      </div>
    )
  },
  {
    title: "Path to Transformers",
    subtitle: "RNN/LSTM Limitations",
    content: (
      <div className="space-y-3">
        <div className="bg-red-900/30 border border-red-600/50 p-3 rounded-xl">
          <h4 className="text-red-400 font-semibold text-sm mb-1">RNN Problems</h4>
          <div className="grid grid-cols-2 gap-2 text-gray-300 text-xs">
            <span>• Sequential — no parallelization</span>
            <span>• Information bottleneck</span>
            <span>• Struggles past ~200 tokens</span>
            <span>• Training: days/weeks</span>
          </div>
        </div>
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h4 className="text-blue-400 font-semibold text-sm">Attention Mechanism (2014)</h4>
          <p className="text-gray-400 text-xs">Dynamically weight input parts — helped but still required RNN</p>
        </div>
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h4 className="text-yellow-400 font-semibold text-sm">Google NMT (2016)</h4>
          <p className="text-gray-300 text-xs">8-layer LSTM • 6 days on 96 K80 GPUs • BLEU: 26.30</p>
        </div>
      </div>
    )
  },
  {
    title: "Attention Is All You Need",
    subtitle: "Vaswani et al., 2017",
    presenterNotes: [
      "Revolutionary paper that introduced transformers",
      "Self-attention: each position attends to all positions",
      "No recurrence needed - full parallelization possible",
      "Multi-head attention: 8 parallel attention operations",
      "10x faster training than RNNs, better BLEU scores"
    ],
    content: (
      <div className="space-y-3">
        <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 p-4 rounded-xl">
          <h4 className="text-white font-semibold text-sm mb-2">Self-Attention</h4>
          <code className="block bg-gray-900 p-2 rounded text-green-300 text-sm text-center">
            Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
          </code>
          <p className="text-gray-300 text-xs text-center mt-2">No recurrence • Full parallelization</p>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <h5 className="text-blue-400 font-semibold text-xs">Multi-Head</h5>
            <p className="text-gray-400 text-xs">8 parallel attention ops</p>
          </div>
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <h5 className="text-purple-400 font-semibold text-xs">Positions</h5>
            <p className="text-gray-400 text-xs">Sin/cos encodings</p>
          </div>
        </div>
        <div className="bg-green-900/30 border border-green-600/50 p-2 rounded-xl">
          <p className="text-green-200 text-sm">3.5 days on 8 P100s (10× faster) • BLEU: 28.4</p>
        </div>
      </div>
    )
  },
  {
    title: "GPT Evolution",
    subtitle: "Decoder-Only Revolution",
    content: (
      <div className="space-y-3">
        <div className="grid grid-cols-3 gap-2">
          <div className="bg-gray-800/50 p-2 rounded-xl text-center">
            <p className="text-blue-400 font-semibold text-xs">GPT (2018)</p>
            <p className="text-xl font-bold text-white">117M</p>
            <p className="text-gray-500 text-xs">BookCorpus</p>
          </div>
          <div className="bg-gray-800/50 p-2 rounded-xl text-center">
            <p className="text-purple-400 font-semibold text-xs">GPT-2 (2019)</p>
            <p className="text-xl font-bold text-white">1.5B</p>
            <p className="text-gray-500 text-xs">Zero-shot</p>
          </div>
          <div className="bg-gray-800/50 p-2 rounded-xl text-center">
            <p className="text-green-400 font-semibold text-xs">GPT-3 (2020)</p>
            <p className="text-xl font-bold text-white">175B</p>
            <p className="text-gray-500 text-xs">In-context</p>
          </div>
        </div>
        <div className="bg-yellow-900/30 border border-yellow-600/50 p-2 rounded-xl">
          <p className="text-yellow-300 font-semibold text-xs mb-1">Why Decoder-Only Won:</p>
          <div className="flex gap-1 text-xs flex-wrap">
            <span className="bg-gray-800 px-2 py-0.5 rounded text-gray-300">Simplicity</span>
            <span className="bg-gray-800 px-2 py-0.5 rounded text-gray-300">Generative</span>
            <span className="bg-gray-800 px-2 py-0.5 rounded text-gray-300">Scaling</span>
            <span className="bg-gray-800 px-2 py-0.5 rounded text-gray-300">Simple loss</span>
            <span className="bg-gray-800 px-2 py-0.5 rounded text-gray-300">Any text</span>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Scaling Laws",
    subtitle: "Kaplan 2020 & Chinchilla 2022",
    content: (
      <div className="space-y-3">
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <code className="block bg-gray-900 p-2 rounded text-green-300 text-sm text-center">
            L(N) = (Nᶜ/N)^αN where αN ≈ 0.076
          </code>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-800/50 p-3 rounded-xl">
            <h4 className="text-purple-400 font-semibold text-sm">Key Findings</h4>
            <ul className="text-gray-300 text-xs">
              <li>• Model size matters most</li>
              <li>• Predictable convergence</li>
            </ul>
          </div>
          <div className="bg-gray-800/50 p-3 rounded-xl">
            <h4 className="text-green-400 font-semibold text-sm">Chinchilla</h4>
            <ul className="text-gray-300 text-xs">
              <li>• 20 tokens per param optimal</li>
              <li>• 70B beats 175B GPT-3</li>
            </ul>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Tensors & Memory",
    subtitle: "Fundamental Data Structures",
    content: (
      <div className="space-y-3">
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h4 className="text-blue-400 font-semibold text-sm mb-1">Tensor Shapes</h4>
          <div className="bg-gray-900 p-2 rounded font-mono text-xs text-green-300">
            <p>embeddings: [batch, seq, d_model] # (32, 512, 768)</p>
            <p>attention: [batch, heads, seq, seq] # (32, 12, 512, 512)</p>
          </div>
        </div>
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h4 className="text-green-400 font-semibold text-sm mb-1">Memory: 7B Model</h4>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="bg-red-900/30 p-2 rounded"><p className="text-red-300 font-bold">28GB</p><p className="text-gray-500 text-xs">FP32</p></div>
            <div className="bg-yellow-900/30 p-2 rounded"><p className="text-yellow-300 font-bold">14GB</p><p className="text-gray-500 text-xs">FP16</p></div>
            <div className="bg-green-900/30 p-2 rounded"><p className="text-green-300 font-bold">7GB</p><p className="text-gray-500 text-xs">INT8</p></div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Transformer Components",
    subtitle: "Modern Architecture",
    content: (
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h4 className="text-blue-400 font-semibold text-sm">RMSNorm</h4>
          <p className="text-gray-400 text-xs">Simpler than LayerNorm, no mean centering</p>
        </div>
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h4 className="text-purple-400 font-semibold text-sm">GQA</h4>
          <p className="text-gray-400 text-xs">Grouped-Query Attention: MHA → MQA → GQA</p>
        </div>
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h4 className="text-green-400 font-semibold text-sm">RoPE</h4>
          <p className="text-gray-400 text-xs">Rotary positions, better extrapolation</p>
        </div>
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h4 className="text-orange-400 font-semibold text-sm">SwiGLU</h4>
          <p className="text-gray-400 text-xs">Gated activation, improved quality</p>
        </div>
      </div>
    )
  },
  {
    title: "Quantization",
    subtitle: "Making Models Fit",
    content: (
      <div className="space-y-3">
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h4 className="text-blue-400 font-semibold text-sm">INT8 (LLM.int8())</h4>
          <div className="grid grid-cols-3 gap-2 text-center text-xs mt-1">
            <span className="text-green-300">2× mem ↓</span>
            <span className="text-yellow-300">10-20% slower</span>
            <span className="text-blue-300">&lt;1% acc loss</span>
          </div>
        </div>
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h4 className="text-purple-400 font-semibold text-sm">QLoRA (4-bit)</h4>
          <div className="flex items-center justify-center gap-1 text-xs mt-1">
            <span className="bg-gray-700 px-2 py-0.5 rounded">FP16</span>
            <span className="text-gray-500">→</span>
            <span className="bg-gray-700 px-2 py-0.5 rounded">NF4</span>
            <span className="text-gray-500">→</span>
            <span className="bg-green-900/50 px-2 py-0.5 rounded text-green-300">5.3× ↓</span>
          </div>
        </div>
        <div className="bg-green-900/30 border border-green-600/50 p-2 rounded-xl">
          <p className="text-gray-300 text-sm">7B: FP16 14GB → <strong className="text-green-300">QLoRA 3-4GB</strong> ✓</p>
        </div>
      </div>
    )
  },
  {
    title: "LoRA",
    subtitle: "Parameter Efficient Fine-tuning",
    presenterNotes: [
      "Low-Rank Adaptation: decompose weight updates into smaller matrices",
      "Only train 0.1-1% of original parameters",
      "128x parameter reduction with minimal quality loss",
      "Perfect for fine-tuning on consumer GPUs",
      "W' = W₀ + BA where r << min(d,k)"
    ],
    content: (
      <div className="space-y-3">
        <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 p-3 rounded-xl">
          <code className="block bg-gray-900 p-2 rounded text-green-300 text-sm text-center">
            W' = W₀ + BA where r ≪ min(d,k)
          </code>
          <p className="text-gray-300 text-xs text-center mt-1">W₀ frozen • B,A trainable low-rank matrices</p>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-red-900/30 p-2 rounded-xl text-center">
            <p className="text-red-300 font-bold">16.7M</p>
            <p className="text-gray-500 text-xs">Original</p>
          </div>
          <div className="bg-green-900/30 p-2 rounded-xl text-center">
            <p className="text-green-300 font-bold">131K</p>
            <p className="text-gray-500 text-xs">LoRA r=16 (128×↓)</p>
          </div>
        </div>
        <div className="bg-yellow-900/30 border border-yellow-600/50 p-2 rounded-xl">
          <p className="text-yellow-200 text-xs">Full 7B: 28GB → LoRA: ~1GB</p>
        </div>
      </div>
    )
  },
  {
    title: "GPU Essentials",
    subtitle: "Hardware for Training",
    content: (
      <div className="space-y-3">
        <table className="w-full text-xs bg-gray-800/50 rounded-xl overflow-hidden">
          <thead><tr className="text-gray-400 border-b border-gray-700">
            <th className="text-left p-2">GPU</th><th className="text-right p-2">BW</th><th className="text-right p-2">TFLOPS</th><th className="text-right p-2">VRAM</th>
          </tr></thead>
          <tbody className="text-gray-300">
            <tr><td className="p-2">A100</td><td className="text-right p-2">1.6TB/s</td><td className="text-right p-2">312</td><td className="text-right p-2">80GB</td></tr>
            <tr><td className="p-2">V100</td><td className="text-right p-2">900GB/s</td><td className="text-right p-2">125</td><td className="text-right p-2">32GB</td></tr>
            <tr><td className="p-2">T4</td><td className="text-right p-2">320GB/s</td><td className="text-right p-2">65</td><td className="text-right p-2">16GB</td></tr>
          </tbody>
        </table>
        <div className="bg-green-900/30 border border-green-600/50 p-2 rounded-xl">
          <p className="text-green-300 font-semibold text-sm">Flash Attention: 2-4× faster, 10-20× less mem</p>
        </div>
      </div>
    )
  },
  {
    title: "Critical Hyperparameters",
    subtitle: "Ranked by Impact",
    presenterNotes: [
      "Learning rate is THE most critical parameter",
      "Too high = NaN loss, too low = no learning",
      "Batch size affects stability and speed",
      "Warmup prevents early instability",
      "LR by model size: <1B=5e-4, 1-7B=1e-4, >7B=5e-5"
    ],
    content: (
      <div className="space-y-3">
        <table className="w-full text-xs bg-gray-800/50 rounded-xl overflow-hidden">
          <thead><tr className="text-gray-400 border-b border-gray-700">
            <th className="text-left p-2">Param</th><th className="text-center p-2">Impact</th><th className="text-right p-2">Range</th>
          </tr></thead>
          <tbody className="text-gray-300">
            <tr><td className="p-2">Learning Rate</td><td className="text-center p-2"><span className="bg-red-900/50 px-1 rounded text-red-300">Critical</span></td><td className="text-right p-2">1e-5 - 1e-3</td></tr>
            <tr><td className="p-2">Batch Size</td><td className="text-center p-2"><span className="bg-orange-900/50 px-1 rounded text-orange-300">High</span></td><td className="text-right p-2">8-512</td></tr>
            <tr><td className="p-2">Warmup</td><td className="text-center p-2"><span className="bg-orange-900/50 px-1 rounded text-orange-300">High</span></td><td className="text-right p-2">3-10%</td></tr>
            <tr><td className="p-2">Weight Decay</td><td className="text-center p-2"><span className="bg-yellow-900/50 px-1 rounded text-yellow-300">Med</span></td><td className="text-right p-2">0.01-0.1</td></tr>
          </tbody>
        </table>
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-blue-400 font-semibold text-xs">LR by size: &lt;1B: 5e-4 | 1-7B: 1e-4 | &gt;7B: 5e-5</p>
        </div>
      </div>
    )
  },
  {
    title: "Model Selection",
    subtitle: "⚠️ The Specialization Trap",
    presenterNotes: [
      "CRITICAL: Specialized models resist learning new knowledge",
      "CodeLlama on new programming language often fails",
      "Base models are flexible clay, specialized are hardened concrete",
      "For novel patterns, always choose base models",
      "Workshop recommendation: Qwen2.5-0.5B or Llama-3.2-1B base"
    ],
    content: (
      <div className="space-y-3">
        <div className="bg-red-900/30 border border-red-600/50 p-2 rounded-xl">
          <p className="text-red-400 font-semibold text-sm">Specialized models resist new knowledge!</p>
          <p className="text-gray-300 text-xs">CodeLlama on new lang fails — base Llama succeeds</p>
        </div>
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <p className="text-blue-400 font-semibold text-sm mb-2">Flexibility Spectrum</p>
          <div className="space-y-1">
            <div className="flex items-center gap-2"><div className="w-36 bg-green-600 h-2 rounded"></div><span className="text-green-300 text-xs">Base (Flexible)</span></div>
            <div className="flex items-center gap-2"><div className="w-24 bg-yellow-600 h-2 rounded"></div><span className="text-yellow-300 text-xs">Continued Pretrain</span></div>
            <div className="flex items-center gap-2"><div className="w-12 bg-orange-600 h-2 rounded"></div><span className="text-orange-300 text-xs">Instruction-Tuned</span></div>
            <div className="flex items-center gap-2"><div className="w-6 bg-red-600 h-2 rounded"></div><span className="text-red-300 text-xs">RLHF Aligned</span></div>
          </div>
        </div>
        <div className="bg-green-900/30 border border-green-600/50 p-2 rounded-xl">
          <p className="text-green-300 text-sm">✓ Qwen2.5-0.5B or Llama-3.2-1B (base)</p>
        </div>
      </div>
    )
  },
  {
    title: "Decision Framework",
    subtitle: "Base vs Specialized",
    content: (
      <div className="space-y-3">
        <table className="w-full text-xs bg-gray-800/50 rounded-xl overflow-hidden">
          <thead><tr className="text-gray-400 border-b border-gray-700">
            <th className="text-left p-2">Task</th><th className="text-center p-2">Wrong</th><th className="text-center p-2">Right</th>
          </tr></thead>
          <tbody className="text-gray-300">
            <tr><td className="p-2">New prog lang</td><td className="text-center p-2 text-red-400">CodeLlama</td><td className="text-center p-2 text-green-400">Base</td></tr>
            <tr><td className="p-2">Python style</td><td className="text-center p-2 text-red-400">Base</td><td className="text-center p-2 text-green-400">CodeLlama</td></tr>
            <tr><td className="p-2">Medical→Legal</td><td className="text-center p-2 text-red-400">BioMedLM</td><td className="text-center p-2 text-green-400">Base</td></tr>
          </tbody>
        </table>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-green-900/30 p-2 rounded-xl">
            <p className="text-green-400 font-semibold text-xs">BASE: New patterns, cross-domain</p>
          </div>
          <div className="bg-blue-900/30 p-2 rounded-xl">
            <p className="text-blue-400 font-semibold text-xs">SPECIALIZED: Extend domain</p>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "MLOps Workflow",
    subtitle: "Complete Pipeline",
    content: (
      <div className="space-y-3">
        <div className="flex items-center justify-center gap-1 text-xs flex-wrap bg-gray-800/50 p-3 rounded-xl">
          <span className="bg-blue-900/50 px-2 py-1 rounded text-blue-200">Config</span>
          <span className="text-gray-500">→</span>
          <span className="bg-blue-900/50 px-2 py-1 rounded text-blue-200">Data</span>
          <span className="text-gray-500">→</span>
          <span className="bg-blue-900/50 px-2 py-1 rounded text-blue-200">Model</span>
          <span className="text-gray-500">→</span>
          <span className="bg-blue-900/50 px-2 py-1 rounded text-blue-200">LoRA</span>
          <span className="text-gray-500">→</span>
          <span className="bg-blue-900/50 px-2 py-1 rounded text-blue-200">Train</span>
          <span className="text-gray-500">→</span>
          <span className="bg-blue-900/50 px-2 py-1 rounded text-blue-200">Eval</span>
          <span className="text-gray-500">→</span>
          <span className="bg-blue-900/50 px-2 py-1 rounded text-blue-200">Save</span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-purple-400 font-semibold text-xs">Hydra Config</p>
            <code className="text-green-300 text-xs">config/model/ training/</code>
          </div>
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-green-400 font-semibold text-xs">W&B Tracking</p>
            <p className="text-gray-400 text-xs">Loss, grads, GPU metrics</p>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Data Pipeline",
    subtitle: "Formats & Quality",
    content: (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-blue-400 font-semibold text-xs">Formats</p>
            <p className="text-gray-400 text-xs">JSONL (simple) • Parquet (fast) • Arrow (zero-copy)</p>
          </div>
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-green-400 font-semibold text-xs">Tokenization</p>
            <p className="text-gray-400 text-xs">On-the-fly • Pre-tokenized • Cached</p>
          </div>
        </div>
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-purple-400 font-semibold text-xs">Quality Checks</p>
          <div className="flex gap-1 mt-1">
            <span className="bg-gray-700 px-1.5 py-0.5 rounded text-gray-300 text-xs">Empty samples</span>
            <span className="bg-gray-700 px-1.5 py-0.5 rounded text-gray-300 text-xs">Token dist</span>
            <span className="bg-gray-700 px-1.5 py-0.5 rounded text-gray-300 text-xs">Duplicates</span>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Debugging Issues",
    subtitle: "Common Problems",
    content: (
      <table className="w-full text-xs bg-gray-800/50 rounded-xl overflow-hidden">
        <thead><tr className="text-gray-400 border-b border-gray-700">
          <th className="text-left p-2">Symptom</th><th className="text-left p-2">Cause</th><th className="text-left p-2">Fix</th>
        </tr></thead>
        <tbody className="text-gray-300">
          <tr><td className="p-2">Loss = NaN</td><td className="p-2">LR too high</td><td className="p-2 text-green-400">LR ÷ 10</td></tr>
          <tr><td className="p-2">Flat loss</td><td className="p-2">LR too low</td><td className="p-2 text-green-400">↑ LR, check data</td></tr>
          <tr><td className="p-2">OOM</td><td className="p-2">Batch size</td><td className="p-2 text-green-400">Grad accum</td></tr>
          <tr><td className="p-2">Spikes</td><td className="p-2">Bad data</td><td className="p-2 text-green-400">Grad clip</td></tr>
          <tr><td className="p-2">Slow</td><td className="p-2">No fp16</td><td className="p-2 text-green-400">Mixed precision</td></tr>
        </tbody>
      </table>
    )
  },
  {
    title: "MLOps Lifecycle",
    subtitle: "Development to Deployment",
    content: (
      <div className="space-y-3">
        <div className="grid grid-cols-6 gap-1 text-center text-xs">
          <div className="p-2 rounded bg-gray-700"><span className="text-gray-300">Experiment</span></div>
          <div className="p-2 rounded bg-gray-700"><span className="text-gray-300">Config</span></div>
          <div className="p-2 rounded bg-gray-700"><span className="text-gray-300">Data</span></div>
          <div className="p-2 rounded bg-green-900/50 border border-green-500"><span className="text-green-300">Train</span></div>
          <div className="p-2 rounded bg-gray-700"><span className="text-gray-300">Eval</span></div>
          <div className="p-2 rounded bg-gray-700"><span className="text-gray-300">Deploy</span></div>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <div className="bg-blue-900/30 p-2 rounded-xl">
            <p className="text-blue-300 font-semibold text-xs">Compute</p>
            <p className="text-gray-400 text-xs">Local → Cloud → Cluster</p>
          </div>
          <div className="bg-purple-900/30 p-2 rounded-xl">
            <p className="text-purple-300 font-semibold text-xs">Storage</p>
            <p className="text-gray-400 text-xs">S3/GCS, checkpoints</p>
          </div>
          <div className="bg-green-900/30 p-2 rounded-xl">
            <p className="text-green-300 font-semibold text-xs">Orchestration</p>
            <p className="text-gray-400 text-xs">Tracking, automation</p>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Tools Ecosystem",
    subtitle: "Hubs & Tracking",
    content: (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-orange-400 font-semibold text-sm">🤗 HuggingFace</p>
            <p className="text-gray-400 text-xs">Models, datasets, APIs</p>
          </div>
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-blue-400 font-semibold text-sm">📊 W&B</p>
            <p className="text-gray-400 text-xs">Metrics, sweeps, artifacts</p>
          </div>
        </div>
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-green-400 font-semibold text-xs mb-1">Alternatives</p>
          <div className="flex gap-1 text-xs">
            <span className="bg-gray-700 px-1.5 py-0.5 rounded">MLflow (open)</span>
            <span className="bg-gray-700 px-1.5 py-0.5 rounded">Neptune</span>
            <span className="bg-gray-700 px-1.5 py-0.5 rounded">TensorBoard</span>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Data Sources",
    subtitle: "Collection & Storage",
    content: (
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <p className="text-blue-400 font-semibold text-sm mb-1">Sources</p>
          <ul className="text-gray-300 text-xs space-y-0.5">
            <li>• Web scraping (Common Crawl)</li>
            <li>• HuggingFace / Kaggle</li>
            <li>• Proprietary data</li>
            <li>• Synthetic generation</li>
          </ul>
        </div>
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <p className="text-green-400 font-semibold text-sm mb-1">Formats</p>
          <ul className="text-gray-300 text-xs space-y-0.5">
            <li><span className="text-yellow-400">JSONL:</span> Simple, slow</li>
            <li><span className="text-green-400">Parquet:</span> Columnar, fast</li>
            <li><span className="text-blue-400">Arrow:</span> Zero-copy</li>
            <li><span className="text-purple-400">WebDataset:</span> Streaming</li>
          </ul>
        </div>
      </div>
    )
  },
  {
    title: "Data Validation",
    subtitle: "Quality Checks",
    content: (
      <div className="space-y-3">
        <div className="grid grid-cols-3 gap-2">
          <div className="bg-red-900/30 p-2 rounded text-center">
            <p className="text-red-300 font-semibold text-sm">Empty</p>
            <p className="text-gray-400 text-xs">Blank samples</p>
          </div>
          <div className="bg-yellow-900/30 p-2 rounded text-center">
            <p className="text-yellow-300 font-semibold text-sm">Length</p>
            <p className="text-gray-400 text-xs">Flag &gt;2048</p>
          </div>
          <div className="bg-orange-900/30 p-2 rounded text-center">
            <p className="text-orange-300 font-semibold text-sm">Dupes</p>
            <p className="text-gray-400 text-xs">Repeats</p>
          </div>
        </div>
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-green-400 font-semibold text-xs mb-1">Cleaning</p>
          <code className="text-green-300 text-xs">normalize whitespace, remove zero-width, NFKC</code>
        </div>
      </div>
    )
  },
  {
    title: "Compute Options",
    subtitle: "Where to Train",
    content: (
      <table className="w-full text-xs bg-gray-800/50 rounded-xl overflow-hidden">
        <thead><tr className="text-gray-400 border-b border-gray-700">
          <th className="text-left p-2">Provider</th><th className="text-left p-2">Pro</th><th className="text-left p-2">Best For</th>
        </tr></thead>
        <tbody className="text-gray-300">
          <tr><td className="p-2 text-blue-300">Local GPU</td><td className="p-2">Control</td><td className="p-2 text-green-400">Dev</td></tr>
          <tr><td className="p-2 text-purple-300">AWS/GCP</td><td className="p-2">Enterprise</td><td className="p-2 text-green-400">Prod</td></tr>
          <tr><td className="p-2 text-orange-300">Lambda/RunPod</td><td className="p-2">Cheap</td><td className="p-2 text-green-400">Training</td></tr>
          <tr><td className="p-2 text-green-300">Colab/Kaggle</td><td className="p-2">Free</td><td className="p-2 text-green-400">Prototype</td></tr>
        </tbody>
      </table>
    )
  },
  {
    title: "Distributed Training",
    subtitle: "Multi-GPU Strategies",
    content: (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-blue-900/30 p-2 rounded-xl">
            <p className="text-blue-300 font-semibold text-sm">DDP</p>
            <p className="text-gray-400 text-xs">Full model per GPU, split batch, average grads</p>
          </div>
          <div className="bg-purple-900/30 p-2 rounded-xl">
            <p className="text-purple-300 font-semibold text-sm">FSDP</p>
            <p className="text-gray-400 text-xs">Shard model across GPUs, for huge models</p>
          </div>
        </div>
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-green-400 font-semibold text-xs">Gradient Accumulation</p>
          <code className="text-green-300 text-xs">effective = micro × accum × gpus</code>
        </div>
      </div>
    )
  },
  {
    title: "Memory Optimization",
    subtitle: "Fitting Models",
    content: (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-blue-400 font-semibold text-xs">Grad Checkpointing</p>
            <p className="text-gray-400 text-xs">30-50% mem ↓, 20-30% slower</p>
          </div>
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-purple-400 font-semibold text-xs">Mixed Precision</p>
            <p className="text-gray-400 text-xs">2× mem ↓, 2-3× faster</p>
          </div>
        </div>
        <div className="bg-green-900/30 border border-green-600/50 p-2 rounded-xl">
          <p className="text-green-300 text-sm font-semibold">QLoRA + GradCkpt + FlashAttn = 7B on 16GB</p>
        </div>
      </div>
    )
  },
  {
    title: "Evaluation",
    subtitle: "Measuring Quality",
    content: (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-blue-400 font-semibold text-xs">Auto Metrics</p>
            <p className="text-gray-400 text-xs">Perplexity, BLEU, ROUGE, Accuracy</p>
          </div>
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-green-400 font-semibold text-xs">Human Eval</p>
            <p className="text-gray-400 text-xs">Quality, safety, A/B tests</p>
          </div>
        </div>
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-purple-400 font-semibold text-xs">Benchmarks</p>
          <div className="flex gap-1 mt-1">
            <span className="bg-gray-700 px-1.5 py-0.5 rounded text-gray-300 text-xs">MMLU</span>
            <span className="bg-gray-700 px-1.5 py-0.5 rounded text-gray-300 text-xs">HumanEval</span>
            <span className="bg-gray-700 px-1.5 py-0.5 rounded text-gray-300 text-xs">BBH</span>
            <span className="bg-gray-700 px-1.5 py-0.5 rounded text-gray-300 text-xs">Custom</span>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Validation",
    subtitle: "Avoiding Overfitting",
    content: (
      <div className="space-y-3">
        <div className="flex gap-2 justify-center">
          <span className="bg-green-900/50 px-2 py-1 rounded text-green-300 text-xs">Train: 80-90%</span>
          <span className="bg-yellow-900/50 px-2 py-1 rounded text-yellow-300 text-xs">Val: 5-10%</span>
          <span className="bg-blue-900/50 px-2 py-1 rounded text-blue-300 text-xs">Test: 5-10%</span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-red-900/30 p-2 rounded-xl">
            <p className="text-red-400 font-semibold text-xs">Overfit Signs</p>
            <p className="text-gray-400 text-xs">Train↓ Val↑, growing gap</p>
          </div>
          <div className="bg-green-900/30 p-2 rounded-xl">
            <p className="text-green-400 font-semibold text-xs">Prevention</p>
            <p className="text-gray-400 text-xs">Early stop, checkpoints</p>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Deployment",
    subtitle: "Inference Frameworks",
    content: (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-blue-400 font-semibold text-xs">Frameworks</p>
            <p className="text-gray-400 text-xs">vLLM • TGI • Triton • llama.cpp</p>
          </div>
          <div className="bg-gray-800/50 p-2 rounded-xl">
            <p className="text-green-400 font-semibold text-xs">Hosting</p>
            <p className="text-gray-400 text-xs">Self • Replicate • Edge • API</p>
          </div>
        </div>
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-purple-400 font-semibold text-xs">Optimizations</p>
          <p className="text-gray-400 text-xs">Continuous batching, KV cache, tensor parallel</p>
        </div>
      </div>
    )
  },
  {
    title: "Scaling Strategies",
    subtitle: "Production Scale",
    content: (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-blue-900/30 p-2 rounded-xl">
            <p className="text-blue-300 font-semibold text-xs">Vertical</p>
            <p className="text-gray-400 text-xs">Bigger GPUs, more VRAM</p>
          </div>
          <div className="bg-purple-900/30 p-2 rounded-xl">
            <p className="text-purple-300 font-semibold text-xs">Horizontal</p>
            <p className="text-gray-400 text-xs">Multiple replicas, load balance</p>
          </div>
        </div>
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-green-400 font-semibold text-xs">Model Optimization</p>
          <p className="text-gray-400 text-xs">Quantization • Distillation • Pruning • Caching</p>
        </div>
      </div>
    )
  },
  {
    title: "Training Script Structure",
    subtitle: "Typical Flow",
    content: (
      <div className="bg-gray-900 p-3 rounded-xl font-mono text-xs text-green-300 space-y-0.5">
        <p className="text-gray-500"># 1. Initialize</p>
        <p>wandb.init(project=cfg.name)</p>
        <p className="text-gray-500"># 2. Load model</p>
        <p>model = AutoModel.from_pretrained(...)</p>
        <p className="text-gray-500"># 3. LoRA config</p>
        <p>model = get_peft_model(model, lora_config)</p>
        <p className="text-gray-500"># 4. Train</p>
        <p>trainer = Trainer(model, args, data)</p>
        <p>trainer.train()</p>
      </div>
    )
  },
  {
    title: "Experiment Tracking",
    subtitle: "What to Log",
    content: (
      <div className="space-y-3">
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <p className="text-blue-400 font-semibold text-sm mb-1">Essential Metrics</p>
          <div className="grid grid-cols-2 gap-1 text-xs">
            <span className="text-gray-300">• train/loss, eval/loss</span>
            <span className="text-gray-300">• learning_rate</span>
            <span className="text-gray-300">• gradient_norm</span>
            <span className="text-gray-300">• GPU memory/util</span>
          </div>
        </div>
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-green-400 font-semibold text-xs">Artifacts</p>
          <p className="text-gray-400 text-xs">Model checkpoints, configs, sample outputs</p>
        </div>
      </div>
    )
  },
  {
    title: "Hyperparameter Sweeps",
    subtitle: "Systematic Search",
    content: (
      <div className="space-y-3">
        <div className="bg-gray-900 p-2 rounded-xl font-mono text-xs text-green-300">
          <p className="text-gray-500"># sweep.yaml</p>
          <p>method: bayes</p>
          <p>metric: eval/loss (minimize)</p>
          <p>parameters:</p>
          <p>  lr: log_uniform [1e-5, 1e-3]</p>
          <p>  lora_r: [8, 16, 32, 64]</p>
        </div>
        <div className="bg-yellow-900/30 border border-yellow-600/50 p-2 rounded-xl">
          <p className="text-yellow-200 text-xs">W&B Sweeps: Bayesian optimization across hyperparams</p>
        </div>
      </div>
    )
  },
  {
    title: "Gradient Debugging",
    subtitle: "Stability Checks",
    content: (
      <div className="space-y-3">
        <div className="bg-gray-900 p-2 rounded-xl font-mono text-xs text-green-300">
          <p>grad_norm = sum(p.grad.norm()**2)**0.5</p>
          <p>if grad_norm &gt; 100:</p>
          <p>  print("⚠️ Large gradients!")</p>
        </div>
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-blue-400 font-semibold text-xs">Key Checks</p>
          <p className="text-gray-400 text-xs">Gradient norms • NaN detection • Loss spikes • LR schedule</p>
        </div>
      </div>
    )
  },
  {
    title: "Profiling",
    subtitle: "Finding Bottlenecks",
    content: (
      <div className="space-y-3">
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-blue-400 font-semibold text-sm">Tools</p>
          <div className="flex gap-1 mt-1">
            <span className="bg-gray-700 px-1.5 py-0.5 rounded text-gray-300 text-xs">PyTorch Profiler</span>
            <span className="bg-gray-700 px-1.5 py-0.5 rounded text-gray-300 text-xs">NVIDIA Nsight</span>
            <span className="bg-gray-700 px-1.5 py-0.5 rounded text-gray-300 text-xs">nvml</span>
          </div>
        </div>
        <div className="bg-gray-800/50 p-2 rounded-xl">
          <p className="text-green-400 font-semibold text-xs">Monitor</p>
          <p className="text-gray-400 text-xs">GPU util, memory, temperature, I/O bottlenecks</p>
        </div>
      </div>
    )
  },
  {
    title: "Key Takeaways",
    subtitle: "Summary",
    presenterNotes: [
      "Transformers replaced RNNs due to parallelization",
      "Decoder-only architecture won due to simplicity",
      "LoRA enables 128x parameter reduction",
      "QLoRA + Flash Attention = massive memory savings",
      "Choose base models for novel patterns",
      "MLOps with Hydra + W&B + monitoring is essential",
      "Workshop focus: Qwen2.5-0.5B or Llama-3.2-1B"
    ],
    content: (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-blue-900/30 p-2 rounded-xl">
            <p className="text-blue-300 font-semibold text-xs">Architecture</p>
            <p className="text-gray-400 text-xs">Transformers replaced RNNs, decoder-only won</p>
          </div>
          <div className="bg-purple-900/30 p-2 rounded-xl">
            <p className="text-purple-300 font-semibold text-xs">Efficiency</p>
            <p className="text-gray-400 text-xs">LoRA 128×, QLoRA 5×, Flash 10×</p>
          </div>
          <div className="bg-green-900/30 p-2 rounded-xl">
            <p className="text-green-300 font-semibold text-xs">Models</p>
            <p className="text-gray-400 text-xs">Base for new patterns, specialized to extend</p>
          </div>
          <div className="bg-orange-900/30 p-2 rounded-xl">
            <p className="text-orange-300 font-semibold text-xs">MLOps</p>
            <p className="text-gray-400 text-xs">Hydra + W&B + monitor everything</p>
          </div>
        </div>
        <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 p-2 rounded-xl text-center">
          <p className="text-white font-semibold text-sm">Workshop: Qwen2.5-0.5B or Llama-3.2-1B (base)</p>
        </div>
      </div>
    )
  }
];

export default function Presentation() {
  const [current, setCurrent] = useState(0);
  
  const next = () => setCurrent(c => Math.min(c + 1, slides.length - 1));
  const prev = () => setCurrent(c => Math.max(c - 1, 0));
  
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'ArrowRight' || e.key === ' ') next();
      if (e.key === 'ArrowLeft') prev();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  const slide = slides[current];
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-4">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-3 text-sm text-gray-400">
          <span>Slide {current + 1} / {slides.length}</span>
          <div className="flex gap-0.5">
            {slides.map((_, i) => (
              <button 
                key={i} 
                onClick={() => setCurrent(i)} 
                className={`w-2 h-2 rounded-full ${i === current ? 'bg-blue-400' : 'bg-gray-600'}`} 
              />
            ))}
          </div>
        </div>
        
        <div className="bg-gray-800/30 rounded-xl p-6 h-[500px] border border-gray-700/50 overflow-y-auto">
          <h2 className="text-2xl font-bold text-white">{slide.title}</h2>
          <p className="text-blue-400 text-sm mb-4">{slide.subtitle}</p>
          <div className="mt-4">
            {slide.content}
          </div>
        </div>
        
        
        {/* Presenter Notes Section */}
        <div className="mt-6 bg-gray-900/50 rounded-xl p-4 border border-gray-600/30">
          <h3 className="text-lg font-semibold text-amber-400 mb-3">📝 Presenter Notes</h3>
          <ul className="space-y-2">
            {(slide.presenterNotes || ['Discuss the key points on this slide', 'Engage with audience questions', 'Transition to next topic']).map((note, index) => (
              <li key={index} className="flex items-start gap-2 text-gray-300">
                <span className="text-amber-400 mt-1">•</span>
                <span className="text-sm leading-relaxed">{note}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="flex justify-between mt-4">
          <button 
            onClick={prev} 
            disabled={current === 0} 
            className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 rounded disabled:opacity-30 hover:bg-gray-600 text-sm"
          >
            <ChevronLeft size={16} /> Prev
          </button>
          <button 
            onClick={next} 
            disabled={current === slides.length - 1} 
            className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 rounded disabled:opacity-30 hover:bg-blue-500 text-sm"
          >
            Next <ChevronRight size={16} />
          </button>
        </div>
        <p className="text-center text-gray-500 text-xs mt-2">← → arrows or spacebar to navigate</p>
      </div>
    </div>
  );
}
