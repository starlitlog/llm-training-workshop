import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const slides = [
  {
    title: "",
    subtitle: "",
    presenterNotes: [
      "Welcome senior software engineers! This isn't theoretical - we're building practical skills",
      "Today: Move from 'AI is magic' to 'I can train and deploy custom models'",
      "Focus on engineering aspects: memory, compute, tooling, infrastructure",
      "By end of session: You'll understand when/how to train vs use existing models",
      "Prerequisites check: Docker, Python, basic understanding of neural networks helpful but not required"
    ],
    content: (
      <div className="space-y-6">
        {/* Hook Section */}
        <div className="text-center mb-8">
          {/* LinkedIn version:
          <div className="bg-red-900/40 border border-red-500/50 rounded-lg p-4 mb-6 inline-block">
            <p className="text-red-300 font-bold text-lg">
              🚨 Most companies are burning $100K+/month on API calls that could cost $50/month
            </p>
          </div>
          */}
          
          <h1 className="text-4xl font-bold text-white mb-3">
            Master LLM Training on Any Budget
          </h1>
          
          <h2 className="text-xl text-blue-300 mb-4">
            How to Train Production-Grade Models for 99% Less Cost
          </h2>
        </div>

        {/* Stats Section */}
        <div className="bg-gray-800/50 rounded-xl p-6 border border-amber-500/30">
          <h3 className="text-center text-amber-300 font-bold text-lg mb-4">
            💰 The Numbers That Will Shock Your CFO
          </h3>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-green-400">$50-500/month</div>
              <div className="text-xs text-gray-400">Custom trained model</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-red-400">$10,000+/month</div>
              <div className="text-xs text-gray-400">OpenAI API at scale</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400">97% savings</div>
              <div className="text-xs text-gray-400">Typical cost reduction</div>
            </div>
          </div>
        </div>

        {/* Value Props */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-emerald-900/30 rounded-xl p-4 border border-emerald-600/30">
            <h3 className="text-lg font-semibold text-emerald-300 mb-3">🎯 What You'll Learn</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>• Train 7B models on $20 consumer GPUs</li>
              <li>• LoRA: 100x less compute, same quality</li>
              <li>• Deploy locally: zero API costs forever</li>
              <li>• Beat GPT-4 on domain-specific tasks</li>
            </ul>
          </div>
          
          <div className="bg-blue-900/30 rounded-xl p-4 border border-blue-600/30">
            <h3 className="text-lg font-semibold text-blue-300 mb-3">⚡ Real Results</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>• Startup saves $120K/year on API costs</li>
              <li>• Custom coding assistant in 2 hours</li>
              <li>• 10x better than GPT on company docs</li>
              <li>• Deploy on laptop, scale to datacenter</li>
            </ul>
          </div>
        </div>

        {/* LLM Pyramid */}
        <div className="bg-gray-800/30 rounded-xl p-5 border border-gray-600/30">
          <h3 className="text-center text-white font-bold text-lg mb-5">The LLM Ecosystem Pyramid</h3>
          
          {/* Level 5 - Top (Easiest/Most Expensive) */}
          <div className="flex justify-center mb-2">
            <div className="bg-red-600/80 rounded-lg px-6 py-2 text-center min-w-[250px] border border-red-400/50">
              <div className="text-white font-semibold text-sm">🏢 Commercial APIs</div>
              <div className="text-red-200 text-xs">OpenAI, Claude, Gemini</div>
              <div className="text-red-300 text-xs">$$$ High cost, zero control</div>
            </div>
          </div>
          
          {/* Level 4 */}
          <div className="flex justify-center mb-2">
            <div className="bg-orange-600/70 rounded-lg px-8 py-2 text-center min-w-[320px] border border-orange-400/50">
              <div className="text-white font-semibold text-sm">🔍 Context Techniques</div>
              <div className="text-orange-200 text-xs">RAG, Vector DBs, Prompt Engineering</div>
              <div className="text-orange-300 text-xs">$$ Medium cost, limited customization</div>
            </div>
          </div>
          
          {/* Level 3 */}
          <div className="flex justify-center mb-2">
            <div className="bg-yellow-600/70 rounded-lg px-10 py-2 text-center min-w-[390px] border border-yellow-400/50">
              <div className="text-white font-semibold text-sm">⚙️ Model Adaptation</div>
              <div className="text-yellow-200 text-xs">Fine-tuning, LoRA, Quantization</div>
              <div className="text-yellow-300 text-xs">$ Lower cost, good customization</div>
            </div>
          </div>
          
          {/* Level 2 - with highlight */}
          <div className="flex justify-center mb-2">
            <div className="bg-green-600/80 rounded-lg px-12 py-2 text-center min-w-[460px] border-4 border-green-400 relative shadow-lg shadow-green-400/50">
              <div className="absolute -top-3 -right-3 bg-amber-500 text-black text-xs font-bold px-2 py-1 rounded-full animate-pulse">
                👉 WE ARE HERE
              </div>
              <div className="text-white font-bold text-sm">🏗️ Base Model Training</div>
              <div className="text-green-200 text-xs">Training from pretrained, Custom architectures</div>
              <div className="text-green-300 text-xs font-semibold">💰 Low cost, full control</div>
            </div>
          </div>
          
          {/* Level 1 - Bottom (Ultimate Foundation) */}
          <div className="flex justify-center mb-3">
            <div className="bg-blue-600/80 rounded-lg px-14 py-3 text-center min-w-[530px] border border-blue-400/50">
              <div className="text-white font-bold text-sm">🧬 Foundation Model Research</div>
              <div className="text-blue-200 text-xs">Creating LLaMA, Gemini, Phi from absolute zero</div>
              <div className="text-blue-300 text-xs">🏛️ Big Tech / Research labs only</div>
            </div>
          </div>
          
          {/* Pyramid explanation */}
          <div className="text-center mt-3">
            <p className="text-gray-400 text-xs">
              <span className="text-amber-300 font-semibold">Most people stay at the top.</span> We're going to level 2 - the sweet spot.
            </p>
          </div>
        </div>

        {/* CTA */}
        <div className="text-center bg-gradient-to-r from-amber-900/40 to-orange-900/40 rounded-xl p-4 border border-amber-500/30">
          <p className="text-amber-200 font-semibold">
            Stop paying API rent. <span className="text-white">Start owning your AI infrastructure.</span>
          </p>
          <p className="text-gray-400 text-sm mt-2">
            From zero to production in one session • No PhD required • Real engineering solutions
          </p>
        </div>
      </div>
    )
  },
  {
    title: "Why LLMs Now? The 80-Year Journey",
    subtitle: "From Theory to Transformers",
    presenterNotes: [
      "For SWEs: Think of this as the evolution from assembly to modern frameworks",
      "1943-1986: 'Assembly era' - basic building blocks, lots of manual work",
      "XOR Problem: The classic example that killed first AI wave - perceptrons couldn't learn XOR function",
      "Hidden layers (1986): Breakthrough that enabled non-linear learning - like adding abstraction layers in software",
      "TensorFlow Playground demo: Show live how adding hidden layer solves XOR (playground.tensorflow.org)",
      "1986-2010: 'High-level languages' - backprop enabled deeper networks",
      "2010-2017: 'Framework era' - ReLU, dropout, batch norm like React/Angular for ML",
      "2017+: 'Modern era' - Transformers like cloud-native, infinitely scalable",
      "Key insight: Each breakthrough solved a fundamental engineering problem"
    ],
    content: (
      <div className="space-y-6">
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-red-900/30 rounded-xl p-4 border border-red-600/30">
            <h3 className="text-lg font-bold text-red-300 mb-2">1943-1986</h3>
            <h4 className="text-sm font-semibold text-red-200 mb-3">Foundation Era</h4>
            <ul className="text-gray-300 text-xs space-y-1">
              <li>• McCulloch-Pitts neuron</li>
              <li>• Perceptron (linear only)</li>
              <li>• XOR problem → AI Winter</li>
              <li>• Expert Systems era</li>
            </ul>
            <div className="mt-3 p-2 bg-red-800/30 rounded text-xs text-red-200">
              Like coding in assembly
            </div>
          </div>
          
          <div className="bg-yellow-900/30 rounded-xl p-4 border border-yellow-600/30">
            <h3 className="text-lg font-bold text-yellow-300 mb-2">1986-2010</h3>
            <h4 className="text-sm font-semibold text-yellow-200 mb-3">Neural Renaissance</h4>
            <ul className="text-gray-300 text-xs space-y-1">
              <li>• Backpropagation (1986)</li>
              <li>• Universal Approximation</li>
              <li>• SVMs dominate</li>
              <li>• Ensemble methods</li>
            </ul>
            <div className="mt-3 p-2 bg-yellow-800/30 rounded text-xs text-yellow-200">
              Like C/C++ era
            </div>
          </div>
          
          <div className="bg-blue-900/30 rounded-xl p-4 border border-blue-600/30">
            <h3 className="text-lg font-bold text-blue-300 mb-2">2010-2017</h3>
            <h4 className="text-sm font-semibold text-blue-200 mb-3">Deep Learning</h4>
            <ul className="text-gray-300 text-xs space-y-1">
              <li>• ReLU fixes gradients</li>
              <li>• Dropout prevents overfit</li>
              <li>• AlexNet breakthrough</li>
              <li>• GPU acceleration</li>
            </ul>
            <div className="mt-3 p-2 bg-blue-800/30 rounded text-xs text-blue-200">
              Like web frameworks
            </div>
          </div>
          
          <div className="bg-green-900/30 rounded-xl p-4 border border-green-600/30">
            <h3 className="text-lg font-bold text-green-300 mb-2">2017+</h3>
            <h4 className="text-sm font-semibold text-green-200 mb-3">Transformer Era</h4>
            <ul className="text-gray-300 text-xs space-y-1">
              <li>• "Attention is All You Need"</li>
              <li>• Parallelizable training</li>
              <li>• Scaling laws discovered</li>
              <li>• Foundation models</li>
            </ul>
            <div className="mt-3 p-2 bg-green-800/30 rounded text-xs text-green-200">
              Like cloud-native
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800/50 rounded-xl p-4">
          <h3 className="text-lg font-semibold text-white mb-3">The Engineering Problems Each Era Solved</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <h4 className="text-blue-400 font-semibold mb-2">Training Issues:</h4>
              <ul className="text-gray-300 space-y-1">
                <li>• <span className="text-red-400">Vanishing gradients</span> → ReLU, normalization</li>
                <li>• <span className="text-red-400">Overfitting</span> → Dropout, regularization</li>
                <li>• <span className="text-red-400">Sequential bottleneck</span> → Attention mechanism</li>
              </ul>
            </div>
            <div>
              <h4 className="text-green-400 font-semibold mb-2">Scale Issues:</h4>
              <ul className="text-gray-300 space-y-1">
                <li>• <span className="text-red-400">Memory limits</span> → Quantization, LoRA</li>
                <li>• <span className="text-red-400">Compute cost</span> → Efficient architectures</li>
                <li>• <span className="text-red-400">Long sequences</span> → Linear attention</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="bg-blue-900/20 border border-blue-600/30 rounded-xl p-4">
          <h3 className="text-lg font-semibold text-blue-300 mb-3">🔬 The XOR Problem & Hidden Layers</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="text-blue-200 font-semibold text-sm">Perceptron Failed (1958):</h4>
              <div className="bg-red-800/30 rounded p-3">
                <p className="text-red-200 text-xs font-mono">
                  XOR Truth Table:<br/>
                  0,0 → 0 ✓<br/>
                  0,1 → 1 ✓<br/>
                  1,0 → 1 ✓<br/>
                  1,1 → 0 ✗
                </p>
                <p className="text-gray-300 text-xs mt-2">Single layer can't learn non-linear patterns</p>
              </div>
            </div>
            <div className="space-y-2">
              <h4 className="text-green-200 font-semibold text-sm">Hidden Layers Solved It (1986):</h4>
              <div className="bg-green-800/30 rounded p-3">
                <p className="text-green-200 text-xs">
                  Hidden layer creates new feature space where XOR becomes linearly separable
                </p>
                <div className="mt-2 p-2 bg-gray-900 rounded">
                  <p className="text-blue-300 text-xs font-mono">
                    🌐 Try it: playground.tensorflow.org
                  </p>
                  <p className="text-gray-400 text-xs mt-1">Interactive demo of neural networks solving XOR</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-3">
          <p className="text-amber-200 text-sm">
            <strong>💡 Key Insight:</strong> Modern LLMs aren't magic - they're the result of solving engineering problems.
            Understanding the history helps you debug when things go wrong.
          </p>
        </div>
      </div>
    )
  },
  {
    title: "Deep Learning Explosion (2010-2015)",
    subtitle: "The Four Pillars That Made Deep Learning Work",
    presenterNotes: [
      "This is when deep learning went from research curiosity to industrial revolution",
      "ReLU: Solved the vanishing gradient problem that plagued deep networks",
      "Dropout: Like having redundant servers - if some fail, others take over",
      "Batch normalization: Like auto-scaling in cloud - keeps things stable as load changes",
      "Adam optimizer: Like having a smart IDE that adjusts coding speed per task",
      "AlexNet 2012: The iPhone moment - proved deep learning could beat everything else",
      "These 4 innovations are still used in every modern model, including LLMs"
    ],
    content: (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-blue-900/20 border border-blue-600/30 rounded-xl p-5">
            <h4 className="text-blue-300 font-semibold text-xl mb-3">🔥 ReLU Activation</h4>
            <div className="space-y-3">
              <div className="bg-blue-800/30 rounded p-3">
                <code className="text-green-300 text-lg font-mono block text-center">f(x) = max(0, x)</code>
              </div>
              <div className="bg-gray-800/50 rounded p-3">
                <h5 className="text-blue-200 font-semibold text-sm mb-2">Why It Was Revolutionary:</h5>
                <ul className="text-gray-300 text-xs space-y-1">
                  <li>• <span className="text-red-400">Before:</span> Sigmoid/tanh gradients → 0 in deep nets</li>
                  <li>• <span className="text-green-400">After:</span> Gradient = 1 for positive values</li>
                  <li>• <span className="text-blue-400">Result:</span> Can train 100+ layer networks</li>
                </ul>
              </div>
              <div className="bg-blue-800/30 rounded p-2">
                <p className="text-blue-200 text-xs"><strong>SWE Analogy:</strong> Like replacing blocking I/O with async - simple change, massive impact</p>
              </div>
            </div>
          </div>
          
          <div className="bg-purple-900/20 border border-purple-600/30 rounded-xl p-5">
            <h4 className="text-purple-300 font-semibold text-xl mb-3">🎲 Dropout (2012)</h4>
            <div className="space-y-3">
              <div className="bg-purple-800/30 rounded p-3">
                <p className="text-purple-200 text-sm text-center">Randomly zero 50% of neurons during training</p>
              </div>
              <div className="bg-gray-800/50 rounded p-3">
                <h5 className="text-purple-200 font-semibold text-sm mb-2">The Problem It Solved:</h5>
                <ul className="text-gray-300 text-xs space-y-1">
                  <li>• <span className="text-red-400">Overfitting:</span> Model memorizes training data</li>
                  <li>• <span className="text-yellow-400">Co-adaptation:</span> Neurons become dependent</li>
                  <li>• <span className="text-green-400">Solution:</span> Force redundancy and robustness</li>
                </ul>
              </div>
              <div className="bg-purple-800/30 rounded p-2">
                <p className="text-purple-200 text-xs"><strong>SWE Analogy:</strong> Like chaos engineering - randomly kill servers to build resilience</p>
              </div>
            </div>
          </div>
          
          <div className="bg-green-900/20 border border-green-600/30 rounded-xl p-5">
            <h4 className="text-green-300 font-semibold text-xl mb-3">⚖️ Batch Normalization</h4>
            <div className="space-y-3">
              <div className="bg-green-800/30 rounded p-3">
                <code className="text-green-200 text-sm block text-center">x̂ = (x - μ) / σ</code>
                <p className="text-gray-300 text-xs text-center mt-1">Normalize inputs to each layer</p>
              </div>
              <div className="bg-gray-800/50 rounded p-3">
                <h5 className="text-green-200 font-semibold text-sm mb-2">Game Changing Benefits:</h5>
                <ul className="text-gray-300 text-xs space-y-1">
                  <li>• <span className="text-blue-400">Higher learning rates:</span> 10× faster training</li>
                  <li>• <span className="text-purple-400">Stable gradients:</span> Less sensitive to init</li>
                  <li>• <span className="text-yellow-400">Regularization effect:</span> Reduces overfitting</li>
                </ul>
              </div>
              <div className="bg-green-800/30 rounded p-2">
                <p className="text-green-200 text-xs"><strong>SWE Analogy:</strong> Like auto-scaling - keeps system stable as load patterns change</p>
              </div>
            </div>
          </div>
          
          <div className="bg-orange-900/20 border border-orange-600/30 rounded-xl p-5">
            <h4 className="text-orange-300 font-semibold text-xl mb-3">🧠 Adam Optimizer</h4>
            <div className="space-y-3">
              <div className="bg-orange-800/30 rounded p-3">
                <p className="text-orange-200 text-sm text-center">Adaptive learning rates per parameter</p>
                <code className="text-green-200 text-xs block text-center mt-1">m̂ / (√v̂ + ε)</code>
              </div>
              <div className="bg-gray-800/50 rounded p-3">
                <h5 className="text-orange-200 font-semibold text-sm mb-2">Why Everyone Uses It:</h5>
                <ul className="text-gray-300 text-xs space-y-1">
                  <li>• <span className="text-blue-400">Momentum:</span> Accelerates in consistent directions</li>
                  <li>• <span className="text-green-400">Adaptation:</span> Slows down in noisy directions</li>
                  <li>• <span className="text-purple-400">Robustness:</span> Works well with minimal tuning</li>
                </ul>
              </div>
              <div className="bg-orange-800/30 rounded p-2">
                <p className="text-orange-200 text-xs"><strong>SWE Analogy:</strong> Like a smart IDE that adjusts typing speed per language/context</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-yellow-900/20 border border-yellow-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-yellow-300 mb-4 text-center">🏆 AlexNet (2012): The Breakthrough Moment</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gray-800/50 rounded p-4 text-center">
              <h4 className="text-yellow-200 font-semibold text-lg">Competition Results</h4>
              <div className="mt-3 space-y-2">
                <div className="bg-red-800/30 rounded p-2">
                  <p className="text-red-300 font-bold text-lg">25.8%</p>
                  <p className="text-gray-400 text-xs">Traditional methods</p>
                </div>
                <div className="bg-green-800/30 rounded p-2">
                  <p className="text-green-300 font-bold text-lg">16.4%</p>
                  <p className="text-gray-400 text-xs">AlexNet (CNNs)</p>
                </div>
                <p className="text-yellow-200 text-sm font-semibold">9.4% gap!</p>
              </div>
            </div>
            
            <div className="bg-gray-800/50 rounded p-4">
              <h4 className="text-yellow-200 font-semibold text-lg mb-2">What Made It Work</h4>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• <span className="text-blue-400">ReLU</span> instead of sigmoid</li>
                <li>• <span className="text-purple-400">Dropout</span> for regularization</li>
                <li>• <span className="text-green-400">GPU training</span> (2 GTX 580s)</li>
                <li>• <span className="text-orange-400">Data augmentation</span></li>
                <li>• <span className="text-red-400">Deep architecture</span> (8 layers)</li>
              </ul>
            </div>
            
            <div className="bg-gray-800/50 rounded p-4">
              <h4 className="text-yellow-200 font-semibold text-lg mb-2">Industry Impact</h4>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• Computer vision revolution</li>
                <li>• Google/Facebook AI investments</li>
                <li>• GPU demand explosion (NVIDIA)</li>
                <li>• Deep learning goes mainstream</li>
                <li>• Path to transformers begins</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-4">
          <h4 className="text-amber-300 font-semibold mb-2">💡 Engineering Impact</h4>
          <p className="text-gray-300 text-sm">
            <strong>These 4 innovations are still the foundation of every modern LLM.</strong>
            ReLU, Dropout, Normalization, and Adam are in GPT-4, Claude, and every model we'll train today.
            Understanding them helps you debug training issues and optimize performance.
          </p>
        </div>
      </div>
    )
  },
  {
    title: "Path to Transformers (2015-2017)",
    subtitle: "Why RNNs Hit a Wall & How Attention Emerged",
    presenterNotes: [
      "This is the critical period - why the old approach couldn't scale",
      "RNNs are like single-threaded programming - fundamentally limited",
      "Attention was initially a 'band-aid' on top of RNNs",
      "The breakthrough was realizing attention could replace RNNs entirely",
      "Google's NMT system shows the peak of what RNNs could achieve",
      "The stage was set for the transformer revolution"
    ],
    content: (
      <div className="space-y-6">
        <div className="bg-red-900/20 border border-red-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-red-300 mb-4 text-center">🚧 The Fundamental Problems with RNNs</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-red-200 font-semibold text-lg">Sequential Processing Bottleneck</h4>
              <div className="bg-red-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-gray-900 rounded p-3">
                    <code className="text-red-300 text-sm font-mono block">
                      for t in range(sequence_length):<br/>
                      &nbsp;&nbsp;h[t] = f(h[t-1], x[t])<br/>
                      &nbsp;&nbsp;# Can't parallelize this!
                    </code>
                  </div>
                  <ul className="text-gray-300 text-xs space-y-1">
                    <li>• <span className="text-red-400">Sequential dependency:</span> Each step needs previous step</li>
                    <li>• <span className="text-red-400">No GPU parallelization:</span> Like single-threaded code</li>
                    <li>• <span className="text-red-400">Training bottleneck:</span> Forward pass is serial</li>
                  </ul>
                </div>
              </div>
              
              <h4 className="text-red-200 font-semibold text-lg">Memory & Gradient Issues</h4>
              <div className="bg-red-800/30 rounded p-4">
                <div className="space-y-2">
                  <div className="bg-red-900/50 rounded p-2 text-center">
                    <p className="text-red-200 text-sm font-semibold">Information Bottleneck</p>
                    <p className="text-gray-300 text-xs">All context must flow through single hidden state</p>
                  </div>
                  <div className="bg-red-900/50 rounded p-2 text-center">
                    <p className="text-red-200 text-sm font-semibold">Vanishing Gradients</p>
                    <p className="text-gray-300 text-xs">Signal degrades over long sequences (&gt;200 tokens)</p>
                  </div>
                  <div className="bg-red-900/50 rounded p-2 text-center">
                    <p className="text-red-200 text-sm font-semibold">Quadratic Memory</p>
                    <p className="text-gray-300 text-xs">LSTM gates require significant memory per timestep</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-yellow-200 font-semibold text-lg">Real-World Impact</h4>
              <div className="bg-yellow-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-gray-800 rounded p-3">
                    <h5 className="text-yellow-200 font-semibold text-sm mb-2">Training Times (2016)</h5>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="text-center">
                        <p className="text-red-300 font-bold text-lg">Days</p>
                        <p className="text-gray-400">Small models</p>
                      </div>
                      <div className="text-center">
                        <p className="text-red-300 font-bold text-lg">Weeks</p>
                        <p className="text-gray-400">Large models</p>
                      </div>
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded p-3">
                    <h5 className="text-yellow-200 font-semibold text-sm mb-2">Hardware Requirements</h5>
                    <ul className="text-gray-300 text-xs space-y-1">
                      <li>• Google: 96 K80 GPUs for translation</li>
                      <li>• Facebook: 100+ GPU clusters</li>
                      <li>• Research labs: Weeks of compute time</li>
                      <li>• Startups: Completely locked out</li>
                    </ul>
                  </div>
                </div>
              </div>
              
              <div className="bg-orange-800/30 rounded p-3">
                <p className="text-orange-200 text-sm font-semibold">🔥 SWE Analogy:</p>
                <p className="text-gray-300 text-xs mt-1">
                  RNNs are like processing a massive dataset with a single-threaded for-loop. 
                  You can optimize the inner loop, but you can't break the fundamental sequential constraint.
                </p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-blue-900/20 border border-blue-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-blue-300 mb-4 text-center">💡 Attention: The First Breakthrough (2014-2016)</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-3">
              <h4 className="text-blue-200 font-semibold text-lg">The Problem It Solved</h4>
              <div className="bg-blue-800/30 rounded p-4">
                <div className="space-y-2">
                  <div className="bg-red-800/50 rounded p-2">
                    <p className="text-red-200 text-xs font-semibold">Before (2014):</p>
                    <p className="text-gray-300 text-xs">Decoder only sees final encoder hidden state</p>
                    <p className="text-gray-300 text-xs">Long sentences lose information</p>
                  </div>
                  <div className="bg-green-800/50 rounded p-2">
                    <p className="text-green-200 text-xs font-semibold">After (Attention):</p>
                    <p className="text-gray-300 text-xs">Decoder can "look at" all encoder states</p>
                    <p className="text-gray-300 text-xs">Weighted combination based on relevance</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="space-y-3">
              <h4 className="text-green-200 font-semibold text-lg">How It Worked</h4>
              <div className="bg-green-800/30 rounded p-4">
                <code className="text-green-300 text-xs font-mono block mb-2">
                  # Bahdanau Attention (2014)<br/>
                  scores = neural_net(decoder_state, encoder_states)<br/>
                  weights = softmax(scores)<br/>
                  context = weighted_sum(encoder_states, weights)
                </code>
                <ul className="text-gray-300 text-xs space-y-1 mt-2">
                  <li>• <span className="text-blue-400">Dynamic weighting:</span> Focus on relevant parts</li>
                  <li>• <span className="text-green-400">No fixed bottleneck:</span> Access all encoder info</li>
                  <li>• <span className="text-purple-400">Learned relevance:</span> NN decides what matters</li>
                </ul>
              </div>
            </div>
            
            <div className="space-y-3">
              <h4 className="text-purple-200 font-semibold text-lg">Breakthrough Results</h4>
              <div className="bg-purple-800/30 rounded p-4">
                <div className="space-y-2 text-center">
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-purple-200 font-semibold text-sm">English → French</p>
                    <div className="grid grid-cols-2 gap-1 mt-1 text-xs">
                      <div>
                        <p className="text-red-300">No Attention</p>
                        <p className="text-red-300 font-bold">BLEU: 30.4</p>
                      </div>
                      <div>
                        <p className="text-green-300">With Attention</p>
                        <p className="text-green-300 font-bold">BLEU: 36.2</p>
                      </div>
                    </div>
                  </div>
                  <p className="text-purple-200 text-xs font-semibold">19% improvement!</p>
                  <ul className="text-gray-300 text-xs space-y-1 mt-2">
                    <li>• Better long sentence handling</li>
                    <li>• Interpretable attention weights</li>
                    <li>• Still required RNN backbone</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800/50 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-white mb-4 text-center">🏔️ The Peak of RNN Era: Google NMT (2016)</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gray-700/50 rounded p-4">
              <h4 className="text-yellow-200 font-semibold text-lg mb-3 text-center">Architecture</h4>
              <div className="space-y-2">
                <div className="bg-yellow-800/30 rounded p-2 text-center">
                  <p className="text-yellow-200 text-sm font-semibold">8-Layer LSTM</p>
                  <p className="text-gray-300 text-xs">Encoder + Decoder</p>
                </div>
                <div className="bg-blue-800/30 rounded p-2 text-center">
                  <p className="text-blue-200 text-sm font-semibold">Attention Mechanism</p>
                  <p className="text-gray-300 text-xs">Luong-style attention</p>
                </div>
                <div className="bg-green-800/30 rounded p-2 text-center">
                  <p className="text-green-200 text-sm font-semibold">Residual Connections</p>
                  <p className="text-gray-300 text-xs">Skip connections for depth</p>
                </div>
                <div className="bg-purple-800/30 rounded p-2 text-center">
                  <p className="text-purple-200 text-sm font-semibold">Beam Search</p>
                  <p className="text-gray-300 text-xs">Sophisticated decoding</p>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-700/50 rounded p-4">
              <h4 className="text-red-200 font-semibold text-lg mb-3 text-center">Resource Requirements</h4>
              <div className="space-y-3">
                <div className="bg-red-800/30 rounded p-3 text-center">
                  <p className="text-red-200 text-xl font-bold">96</p>
                  <p className="text-gray-300 text-xs">NVIDIA K80 GPUs</p>
                  <p className="text-gray-400 text-xs mt-1">($300K+ in hardware)</p>
                </div>
                <div className="bg-orange-800/30 rounded p-3 text-center">
                  <p className="text-orange-200 text-xl font-bold">6</p>
                  <p className="text-gray-300 text-xs">Days of training</p>
                  <p className="text-gray-400 text-xs mt-1">(144 GPU-days total)</p>
                </div>
                <div className="bg-yellow-800/30 rounded p-3 text-center">
                  <p className="text-yellow-200 text-xl font-bold">$50K+</p>
                  <p className="text-gray-300 text-xs">Estimated compute cost</p>
                  <p className="text-gray-400 text-xs mt-1">(2016 cloud pricing)</p>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-700/50 rounded p-4">
              <h4 className="text-green-200 font-semibold text-lg mb-3 text-center">Performance</h4>
              <div className="space-y-3">
                <div className="bg-green-800/30 rounded p-3 text-center">
                  <p className="text-green-200 text-xl font-bold">26.30</p>
                  <p className="text-gray-300 text-xs">BLEU Score (EN-DE)</p>
                  <p className="text-gray-400 text-xs mt-1">State-of-the-art 2016</p>
                </div>
                <div className="bg-blue-800/30 rounded p-3">
                  <p className="text-blue-200 text-sm font-semibold mb-1">Breakthrough Features:</p>
                  <ul className="text-gray-300 text-xs space-y-1">
                    <li>• Zero-shot translation</li>
                    <li>• Multilingual capability</li>
                    <li>• Production deployment</li>
                    <li>• Sub-word tokenization</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-4 bg-amber-800/30 rounded p-3 text-center">
            <p className="text-amber-200 text-lg font-semibold">
              🎯 This represented the absolute peak of what RNNs could achieve
            </p>
            <p className="text-gray-300 text-sm mt-1">
              Any further progress would require a fundamentally different architecture...
            </p>
          </div>
        </div>
        
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-4">
          <h4 className="text-amber-300 font-semibold mb-2">💡 Engineering Insight</h4>
          <p className="text-gray-300 text-sm">
            <strong>The stage was perfectly set for transformers:</strong> RNNs had hit fundamental scaling limits, 
            attention had proven its value, and the industry needed massive parallelization. 
            The next breakthrough would eliminate RNNs entirely and make attention the star.
          </p>
        </div>
      </div>
    )
  },
  {
    title: "The Transformer Revolution",
    subtitle: "\"Attention Is All You Need\" - Vaswani et al., 2017",
    presenterNotes: [
      "This is THE paper that changed everything - like the iPhone moment for AI",
      "Before: RNNs were sequential, slow to train, couldn't capture long dependencies",
      "After: Transformers are parallelizable, scalable, and handle any sequence length",
      "Key insight: Attention mechanism can replace recurrence entirely",
      "Think of it like moving from synchronous to async programming",
      "Self-attention = each token can look at all other tokens simultaneously"
    ],
    content: (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-red-900/20 border border-red-600/30 rounded-xl p-4">
            <h3 className="text-lg font-semibold text-red-300 mb-3">🐌 Before: RNN/LSTM Problems</h3>
            <div className="space-y-3">
              <div className="bg-red-800/30 rounded p-3">
                <h4 className="text-red-200 font-semibold text-sm">Sequential Processing</h4>
                <p className="text-gray-300 text-xs mt-1">Like processing array with for-loop - can't parallelize</p>
                <code className="block text-xs text-red-200 mt-1 font-mono">for token in sequence: process(token)</code>
              </div>
              <div className="bg-red-800/30 rounded p-3">
                <h4 className="text-red-200 font-semibold text-sm">Information Bottleneck</h4>
                <p className="text-gray-300 text-xs mt-1">All info must flow through hidden state</p>
                <p className="text-red-200 text-xs">Forgets start of long sequences</p>
              </div>
              <div className="text-center text-red-300 text-sm font-mono">
                Training time: weeks on Google's hardware
              </div>
            </div>
          </div>
          
          <div className="bg-green-900/20 border border-green-600/30 rounded-xl p-4">
            <h3 className="text-lg font-semibold text-green-300 mb-3">⚡ After: Transformer Solution</h3>
            <div className="space-y-3">
              <div className="bg-green-800/30 rounded p-3">
                <h4 className="text-green-200 font-semibold text-sm">Parallel Processing</h4>
                <p className="text-gray-300 text-xs mt-1">Like GPU compute - all tokens process simultaneously</p>
                <code className="block text-xs text-green-200 mt-1 font-mono">parallel_map(attention, tokens)</code>
              </div>
              <div className="bg-green-800/30 rounded p-3">
                <h4 className="text-green-200 font-semibold text-sm">Direct Attention</h4>
                <p className="text-gray-300 text-xs mt-1">Every token can directly attend to every other</p>
                <p className="text-green-200 text-xs">No information loss</p>
              </div>
              <div className="text-center text-green-300 text-sm font-mono">
                Training time: 3.5 days on 8 GPUs
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-6 border border-blue-600/30">
          <h3 className="text-xl font-semibold text-white mb-4 text-center">The Attention Mechanism</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="bg-blue-800/50 rounded p-3 mb-2">
                <h4 className="text-blue-200 font-semibold">Query (Q)</h4>
                <p className="text-xs text-gray-300 mt-1">"What am I looking for?"</p>
              </div>
            </div>
            <div className="text-center">
              <div className="bg-purple-800/50 rounded p-3 mb-2">
                <h4 className="text-purple-200 font-semibold">Key (K)</h4>
                <p className="text-xs text-gray-300 mt-1">"What do I offer?"</p>
              </div>
            </div>
            <div className="text-center">
              <div className="bg-green-800/50 rounded p-3 mb-2">
                <h4 className="text-green-200 font-semibold">Value (V)</h4>
                <p className="text-xs text-gray-300 mt-1">"Here's my content"</p>
              </div>
            </div>
          </div>
          <div className="mt-4 bg-gray-900/50 rounded-lg p-4 text-center">
            <code className="text-green-300 text-lg font-mono">
              Attention(Q,K,V) = softmax(QK<sup>T</sup>/√d<sub>k</sub>)V
            </code>
            <p className="text-gray-300 text-sm mt-2">Like a database join where similarity determines weight</p>
          </div>
        </div>
        
        <div className="bg-blue-900/20 border border-blue-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-blue-300 mb-4 text-center">🔧 Multi-Head Attention: The Core Innovation</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-3">
              <h4 className="text-blue-200 font-semibold text-lg">Why "Multi-Head"?</h4>
              <div className="bg-blue-800/30 rounded p-3">
                <p className="text-blue-200 text-sm mb-2">8 parallel attention operations</p>
                <ul className="text-gray-300 text-xs space-y-1">
                  <li>• Each head learns different relationships</li>
                  <li>• Head 1: Syntax (subject-verb)</li>
                  <li>• Head 2: Semantics (word meaning)</li>
                  <li>• Head 3: Coreference (pronouns)</li>
                  <li>• Etc...</li>
                </ul>
              </div>
            </div>
            <div className="space-y-3">
              <h4 className="text-green-200 font-semibold text-lg">Computational Magic</h4>
              <div className="bg-green-800/30 rounded p-3">
                <p className="text-green-200 text-sm mb-2">Matrix multiplication parallelization</p>
                <code className="block text-xs text-green-300 font-mono">
                  QK^T: [batch, seq, d] @ [batch, d, seq]<br/>
                  Result: [batch, seq, seq]<br/>
                  GPU optimized!
                </code>
              </div>
            </div>
            <div className="space-y-3">
              <h4 className="text-purple-200 font-semibold text-lg">Position Encoding</h4>
              <div className="bg-purple-800/30 rounded p-3">
                <p className="text-purple-200 text-sm mb-2">No built-in sequence order</p>
                <code className="block text-xs text-purple-300 font-mono">
                  PE(pos,2i) = sin(pos/10000^(2i/d))<br/>
                  PE(pos,2i+1) = cos(pos/10000^(2i/d))
                </code>
                <p className="text-gray-300 text-xs mt-1">Adds positional information via sin/cos waves</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800/50 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-white mb-4 text-center">🚀 Performance Revolution: The Numbers</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-red-300 font-semibold text-lg">Google's NMT System (2016)</h4>
              <div className="bg-red-800/30 rounded p-4">
                <div className="grid grid-cols-2 gap-3 text-center">
                  <div>
                    <p className="text-red-200 font-bold text-2xl">96</p>
                    <p className="text-gray-400 text-xs">K80 GPUs needed</p>
                  </div>
                  <div>
                    <p className="text-red-200 font-bold text-2xl">6</p>
                    <p className="text-gray-400 text-xs">Days of training</p>
                  </div>
                </div>
                <div className="mt-3 p-2 bg-red-900/50 rounded text-center">
                  <p className="text-red-200 text-sm font-semibold">BLEU Score: 26.30</p>
                </div>
                <ul className="text-gray-300 text-xs mt-3 space-y-1">
                  <li>• 8-layer LSTM with attention</li>
                  <li>• Sequential processing bottleneck</li>
                  <li>• Millions in compute costs</li>
                </ul>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-green-300 font-semibold text-lg">Transformer (2017)</h4>
              <div className="bg-green-800/30 rounded p-4">
                <div className="grid grid-cols-2 gap-3 text-center">
                  <div>
                    <p className="text-green-200 font-bold text-2xl">8</p>
                    <p className="text-gray-400 text-xs">P100 GPUs needed</p>
                  </div>
                  <div>
                    <p className="text-green-200 font-bold text-2xl">3.5</p>
                    <p className="text-gray-400 text-xs">Days of training</p>
                  </div>
                </div>
                <div className="mt-3 p-2 bg-green-900/50 rounded text-center">
                  <p className="text-green-200 text-sm font-semibold">BLEU Score: 28.4</p>
                </div>
                <ul className="text-gray-300 text-xs mt-3 space-y-1">
                  <li>• 6-layer encoder + 6-layer decoder</li>
                  <li>• Fully parallelizable</li>
                  <li>• 12× fewer GPUs, better results!</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="mt-4 bg-amber-800/30 rounded p-3 text-center">
            <p className="text-amber-200 text-lg font-semibold">
              🎯 Result: 12× fewer resources, 7% better performance, infinitely more scalable
            </p>
          </div>
        </div>
        
        <div className="bg-purple-900/20 border border-purple-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-purple-300 mb-4 text-center">⚡ Why Transformers Scale Like Nothing Before</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-purple-800/30 rounded p-4">
              <h4 className="text-purple-200 font-semibold text-lg mb-2">Computational Complexity</h4>
              <div className="space-y-2 text-sm">
                <div className="bg-red-800/50 rounded p-2">
                  <p className="text-red-200 font-semibold">RNN/LSTM:</p>
                  <code className="text-red-300 text-xs">O(n × d²) serial</code>
                  <p className="text-gray-300 text-xs">Can't parallelize sequence</p>
                </div>
                <div className="bg-green-800/50 rounded p-2">
                  <p className="text-green-200 font-semibold">Transformer:</p>
                  <code className="text-green-300 text-xs">O(n² × d) parallel</code>
                  <p className="text-gray-300 text-xs">All positions at once</p>
                </div>
              </div>
            </div>
            
            <div className="bg-blue-800/30 rounded p-4">
              <h4 className="text-blue-200 font-semibold text-lg mb-2">Path Length</h4>
              <div className="space-y-2 text-sm">
                <div className="bg-red-800/50 rounded p-2 text-center">
                  <p className="text-red-200 font-semibold">RNN:</p>
                  <p className="text-red-300 text-xl font-bold">O(n)</p>
                  <p className="text-gray-300 text-xs">Token 1 → Token n</p>
                  <p className="text-gray-300 text-xs">Long gradient paths</p>
                </div>
                <div className="bg-green-800/50 rounded p-2 text-center">
                  <p className="text-green-200 font-semibold">Transformer:</p>
                  <p className="text-green-300 text-xl font-bold">O(1)</p>
                  <p className="text-gray-300 text-xs">Direct connections</p>
                  <p className="text-gray-300 text-xs">No gradient decay</p>
                </div>
              </div>
            </div>
            
            <div className="bg-orange-800/30 rounded p-4">
              <h4 className="text-orange-200 font-semibold text-lg mb-2">Memory Patterns</h4>
              <div className="space-y-2 text-sm">
                <div className="bg-red-800/50 rounded p-2">
                  <p className="text-red-200 font-semibold mb-1">RNN Memory:</p>
                  <p className="text-gray-300 text-xs">Hidden state bottleneck</p>
                  <p className="text-gray-300 text-xs">Forgets long sequences</p>
                </div>
                <div className="bg-green-800/50 rounded p-2">
                  <p className="text-green-200 font-semibold mb-1">Transformer:</p>
                  <p className="text-gray-300 text-xs">Full attention matrix</p>
                  <p className="text-gray-300 text-xs">Perfect memory access</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-4">
          <h4 className="text-amber-300 font-semibold mb-2">💡 Engineering Insight</h4>
          <p className="text-gray-300 text-sm">
            <strong>Transformers solved the fundamental scalability problem of sequence modeling.</strong>
            Think RNNs = single-threaded processing vs Transformers = massively parallel MapReduce for language.
            This architectural breakthrough enabled everything from GPT to Claude.
          </p>
        </div>
      </div>
    )
  },
  {
    title: "GPT Evolution: The Decoder-Only Revolution",
    subtitle: "From Proof of Concept to AGI Contender (2018-2023)",
    presenterNotes: [
      "This is the most important architectural decision in AI history",
      "Encoder-Decoder seemed obvious (like BERT), but decoder-only won completely",
      "Each GPT generation showed exponential improvement, not just linear scaling",
      "GPT-1: Proved the concept, GPT-2: Showed emergence, GPT-3: Changed everything",
      "The 'too dangerous to release' moment was when we realized this could scale indefinitely",
      "Decoder-only architecture is what every major LLM uses now: GPT-4, Claude, Llama, etc."
    ],
    content: (
      <div className="space-y-6">
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-amber-300 mb-4 text-center">🤔 The Architecture Decision That Changed Everything</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-red-200 font-semibold text-lg">Why Not Encoder-Decoder?</h4>
              <div className="bg-red-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-gray-900 rounded p-3">
                    <h5 className="text-red-200 font-semibold text-sm mb-2">BERT-style (2018)</h5>
                    <ul className="text-gray-300 text-xs space-y-1">
                      <li>• <span className="text-red-400">Bidirectional:</span> Can see future tokens</li>
                      <li>• <span className="text-red-400">Complex:</span> Encoder + decoder stacks</li>
                      <li>• <span className="text-red-400">Task-specific:</span> Need fine-tuning for each use</li>
                      <li>• <span className="text-red-400">Generation:</span> Awkward for text generation</li>
                    </ul>
                  </div>
                  <div className="bg-red-900/50 rounded p-2 text-center">
                    <p className="text-red-200 text-xs">Great for understanding, poor for generation</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-green-200 font-semibold text-lg">Why Decoder-Only Won</h4>
              <div className="bg-green-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-gray-900 rounded p-3">
                    <h5 className="text-green-200 font-semibold text-sm mb-2">GPT-style (2018+)</h5>
                    <ul className="text-gray-300 text-xs space-y-1">
                      <li>• <span className="text-green-400">Causal:</span> Only sees past tokens (like humans)</li>
                      <li>• <span className="text-green-400">Simple:</span> Just decoder stack</li>
                      <li>• <span className="text-green-400">Universal:</span> One model, many tasks</li>
                      <li>• <span className="text-green-400">Natural:</span> Perfect for text generation</li>
                    </ul>
                  </div>
                  <div className="bg-green-900/50 rounded p-2 text-center">
                    <p className="text-green-200 text-xs">Excellent for both understanding AND generation</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-4 bg-blue-800/30 rounded p-3 text-center">
            <p className="text-blue-200 text-sm font-semibold">
              🎯 Key Insight: Language is fundamentally sequential - humans read left-to-right, predict next words
            </p>
          </div>
        </div>
        
        <div className="bg-gray-800/50 rounded-xl p-6">
          <h3 className="text-2xl font-semibold text-white mb-6 text-center">📈 The Exponential Journey</h3>
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-blue-900/30 rounded-xl p-4 border border-blue-600/30">
              <h4 className="text-blue-300 font-bold text-xl mb-3 text-center">GPT-1 (2018)</h4>
              <div className="space-y-3">
                <div className="bg-blue-800/50 rounded p-3 text-center">
                  <p className="text-blue-200 text-2xl font-bold">117M</p>
                  <p className="text-gray-400 text-xs">Parameters</p>
                </div>
                <div className="space-y-2 text-xs">
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-blue-200 font-semibold">Breakthrough:</p>
                    <p className="text-gray-300">Unsupervised pretraining works!</p>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-blue-200 font-semibold">Architecture:</p>
                    <p className="text-gray-300">12 layers, 768 hidden</p>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-blue-200 font-semibold">Training:</p>
                    <p className="text-gray-300">BooksCorpus (~4GB text)</p>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-blue-200 font-semibold">Impact:</p>
                    <p className="text-gray-300">Proof of concept</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-purple-900/30 rounded-xl p-4 border border-purple-600/30">
              <h4 className="text-purple-300 font-bold text-xl mb-3 text-center">GPT-2 (2019)</h4>
              <div className="space-y-3">
                <div className="bg-purple-800/50 rounded p-3 text-center">
                  <p className="text-purple-200 text-2xl font-bold">1.5B</p>
                  <p className="text-gray-400 text-xs">Parameters</p>
                </div>
                <div className="space-y-2 text-xs">
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-purple-200 font-semibold">Breakthrough:</p>
                    <p className="text-gray-300">Emergent capabilities appear</p>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-purple-200 font-semibold">Architecture:</p>
                    <p className="text-gray-300">48 layers, 1600 hidden</p>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-purple-200 font-semibold">Training:</p>
                    <p className="text-gray-300">WebText (~40GB text)</p>
                  </div>
                  <div className="bg-red-800/30 rounded p-2">
                    <p className="text-red-200 font-semibold">"Too Dangerous"</p>
                    <p className="text-gray-300">Initially withheld</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-green-900/30 rounded-xl p-4 border border-green-600/30">
              <h4 className="text-green-300 font-bold text-xl mb-3 text-center">GPT-3 (2020)</h4>
              <div className="space-y-3">
                <div className="bg-green-800/50 rounded p-3 text-center">
                  <p className="text-green-200 text-2xl font-bold">175B</p>
                  <p className="text-gray-400 text-xs">Parameters</p>
                </div>
                <div className="space-y-2 text-xs">
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-green-200 font-semibold">Breakthrough:</p>
                    <p className="text-gray-300">Few-shot learning!</p>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-green-200 font-semibold">Architecture:</p>
                    <p className="text-gray-300">96 layers, 12k hidden</p>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-green-200 font-semibold">Training:</p>
                    <p className="text-gray-300">CommonCrawl (~570GB)</p>
                  </div>
                  <div className="bg-orange-800/30 rounded p-2">
                    <p className="text-orange-200 font-semibold">Cost:</p>
                    <p className="text-gray-300">$12M to train</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-amber-900/30 rounded-xl p-4 border border-amber-600/30">
              <h4 className="text-amber-300 font-bold text-xl mb-3 text-center">GPT-4 (2023)</h4>
              <div className="space-y-3">
                <div className="bg-amber-800/50 rounded p-3 text-center">
                  <p className="text-amber-200 text-2xl font-bold">~1.8T</p>
                  <p className="text-gray-400 text-xs">Parameters*</p>
                </div>
                <div className="space-y-2 text-xs">
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-amber-200 font-semibold">Breakthrough:</p>
                    <p className="text-gray-300">Multimodal reasoning</p>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-amber-200 font-semibold">Architecture:</p>
                    <p className="text-gray-300">Mixture of Experts</p>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <p className="text-amber-200 font-semibold">Training:</p>
                    <p className="text-gray-300">Internet-scale data</p>
                  </div>
                  <div className="bg-red-800/30 rounded p-2">
                    <p className="text-red-200 font-semibold">Cost:</p>
                    <p className="text-gray-300">$100M+ estimated</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-6 bg-blue-800/30 rounded p-4">
            <h4 className="text-blue-300 font-semibold text-lg mb-3 text-center">🚀 The Scaling Laws</h4>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-blue-200 font-semibold text-lg">Parameters</p>
                <p className="text-green-300 text-sm">117M → 1.8T</p>
                <p className="text-gray-400 text-xs">15,384× increase</p>
              </div>
              <div>
                <p className="text-purple-200 font-semibold text-lg">Training Data</p>
                <p className="text-green-300 text-sm">4GB → 10TB+</p>
                <p className="text-gray-400 text-xs">2,500× increase</p>
              </div>
              <div>
                <p className="text-orange-200 font-semibold text-lg">Training Cost</p>
                <p className="text-green-300 text-sm">$1K → $100M+</p>
                <p className="text-gray-400 text-xs">100,000× increase</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-purple-900/20 border border-purple-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-purple-300 mb-4 text-center">🧠 Emergent Capabilities: The Phase Transitions</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-purple-200 font-semibold text-lg">What "Emergence" Means</h4>
              <div className="bg-purple-800/30 rounded p-4">
                <p className="text-purple-200 text-sm mb-3 font-semibold">Capabilities that suddenly appear at scale:</p>
                <ul className="text-gray-300 text-xs space-y-2">
                  <li>• <span className="text-blue-400">GPT-1:</span> Basic language coherence</li>
                  <li>• <span className="text-purple-400">GPT-2:</span> Creative writing, simple reasoning</li>
                  <li>• <span className="text-green-400">GPT-3:</span> Few-shot learning, code generation</li>
                  <li>• <span className="text-amber-400">GPT-4:</span> Advanced reasoning, multimodal understanding</li>
                </ul>
                <div className="mt-3 bg-gray-800 rounded p-2">
                  <p className="text-purple-200 text-xs font-semibold">🔬 SWE Analogy:</p>
                  <p className="text-gray-300 text-xs">Like distributed systems - add enough nodes and new properties emerge (fault tolerance, consensus, etc.)</p>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-green-200 font-semibold text-lg">The "Too Dangerous" Moment</h4>
              <div className="bg-green-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-red-800/50 rounded p-3">
                    <h5 className="text-red-200 font-semibold text-sm mb-2">GPT-2 Could:</h5>
                    <ul className="text-gray-300 text-xs space-y-1">
                      <li>• Write convincing fake news</li>
                      <li>• Complete articles in any style</li>
                      <li>• Generate coherent long-form text</li>
                      <li>• Adapt to prompts dynamically</li>
                    </ul>
                  </div>
                  <div className="bg-yellow-800/50 rounded p-3">
                    <h5 className="text-yellow-200 font-semibold text-sm mb-2">OpenAI's Dilemma:</h5>
                    <ul className="text-gray-300 text-xs space-y-1">
                      <li>• First time AI was "too good"</li>
                      <li>• Staged release strategy</li>
                      <li>• Set precedent for responsible AI</li>
                      <li>• Showed scaling could be dangerous</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-4">
          <h4 className="text-amber-300 font-semibold mb-2">💡 Engineering Insight</h4>
          <p className="text-gray-300 text-sm">
            <strong>Decoder-only architecture won because it mirrors how humans process language:</strong> sequential, causal, predictive. 
            Every major LLM today (GPT-4, Claude, Llama, Gemini) uses this architecture. 
            Understanding this pattern helps you choose the right base model and predict how capabilities will scale.
          </p>
        </div>
      </div>
    )
  },
  {
    title: "Scaling Laws & Memory: The Engineering Reality",
    subtitle: "From Theory to Hardware Constraints",
    presenterNotes: [
      "This combines the mathematical theory with practical engineering constraints",
      "Scaling laws predict performance, but memory constraints dictate what you can actually run",
      "Chinchilla discovery: Most models are undertrained, not oversized",
      "Understanding tensor shapes helps debug memory issues and optimize performance",
      "These numbers directly impact your hardware choices and training strategies"
    ],
    content: (
      <div className="space-y-6">
        <div className="bg-blue-900/20 border border-blue-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-blue-300 mb-4 text-center">📐 Scaling Laws: The Predictable Rules</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-blue-200 font-semibold text-lg">Kaplan et al. (2020)</h4>
              <div className="bg-blue-800/30 rounded p-4">
                <div className="bg-gray-900 rounded p-3 mb-3">
                  <code className="text-green-300 text-lg font-mono block text-center">
                    L(N) = (N<sub>c</sub>/N)<sup>α</sup>
                  </code>
                  <p className="text-gray-300 text-xs text-center mt-1">where α ≈ 0.076</p>
                </div>
                <h5 className="text-blue-200 font-semibold text-sm mb-2">Key Discoveries:</h5>
                <ul className="text-gray-300 text-xs space-y-1">
                  <li>• <span className="text-green-400">Model size</span> matters most for performance</li>
                  <li>• <span className="text-blue-400">Predictable convergence</span> - no surprises</li>
                  <li>• <span className="text-purple-400">Power law relationship</span> - consistent scaling</li>
                  <li>• <span className="text-yellow-400">Compute optimal</span> - bigger is usually better</li>
                </ul>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-green-200 font-semibold text-lg">Chinchilla (2022) - The Game Changer</h4>
              <div className="bg-green-800/30 rounded p-4">
                <div className="bg-red-800/30 rounded p-3 mb-3">
                  <p className="text-red-200 text-sm font-semibold mb-1">Previous Wisdom (Wrong!):</p>
                  <p className="text-gray-300 text-xs">"Bigger models = better, regardless of training data"</p>
                </div>
                <div className="bg-green-800/50 rounded p-3 mb-3">
                  <p className="text-green-200 text-sm font-semibold mb-1">Chinchilla Discovery:</p>
                  <p className="text-gray-300 text-xs"><strong>20 tokens per parameter</strong> is optimal</p>
                </div>
                <h5 className="text-green-200 font-semibold text-sm mb-2">Revolutionary Results:</h5>
                <ul className="text-gray-300 text-xs space-y-1">
                  <li>• <span className="text-green-400">70B Chinchilla</span> beats 175B GPT-3</li>
                  <li>• <span className="text-blue-400">4× less compute</span> for same performance</li>
                  <li>• <span className="text-purple-400">Most models are undertrained</span> not oversized</li>
                  <li>• <span className="text-yellow-400">Data quality &gt; model size</span></li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="mt-4 bg-amber-800/30 rounded p-3 text-center">
            <p className="text-amber-200 text-sm font-semibold">
              💡 Practical Impact: Train smaller models longer, not bigger models briefly
            </p>
          </div>
        </div>
        
        <div className="bg-purple-900/20 border border-purple-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-purple-300 mb-4 text-center">🔢 Tensor Shapes: The Memory Reality</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-purple-200 font-semibold text-lg">Understanding Tensor Dimensions</h4>
              <div className="bg-purple-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-gray-900 rounded p-3">
                    <h5 className="text-purple-200 font-semibold text-sm mb-2">Core Tensors:</h5>
                    <div className="font-mono text-xs space-y-1">
                      <p className="text-green-300">embeddings: [batch, seq, d_model]</p>
                      <p className="text-gray-400 ml-4"># (32, 512, 768) = 12.6M floats</p>
                      <p className="text-blue-300">attention: [batch, heads, seq, seq]</p>
                      <p className="text-gray-400 ml-4"># (32, 12, 512, 512) = 100M floats</p>
                      <p className="text-yellow-300">weights: [d_model, d_model]</p>
                      <p className="text-gray-400 ml-4"># (768, 768) = 590K floats</p>
                    </div>
                  </div>
                  <div className="bg-purple-800/50 rounded p-2">
                    <p className="text-purple-200 text-xs font-semibold">🔧 SWE Analogy:</p>
                    <p className="text-gray-300 text-xs">Like multi-dimensional arrays in your favorite language, but optimized for parallel GPU operations</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-orange-200 font-semibold text-lg">Memory Requirements by Model Size</h4>
              <div className="bg-orange-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-gray-800 rounded p-3">
                    <h5 className="text-orange-200 font-semibold text-sm mb-2">7B Model Storage:</h5>
                    <div className="grid grid-cols-3 gap-2 text-center">
                      <div className="bg-red-800/30 p-2 rounded">
                        <p className="text-red-300 font-bold text-lg">28GB</p>
                        <p className="text-gray-400 text-xs">FP32</p>
                        <p className="text-gray-400 text-xs">4 bytes/param</p>
                      </div>
                      <div className="bg-yellow-800/30 p-2 rounded">
                        <p className="text-yellow-300 font-bold text-lg">14GB</p>
                        <p className="text-gray-400 text-xs">FP16</p>
                        <p className="text-gray-400 text-xs">2 bytes/param</p>
                      </div>
                      <div className="bg-green-800/30 p-2 rounded">
                        <p className="text-green-300 font-bold text-lg">7GB</p>
                        <p className="text-gray-400 text-xs">INT8</p>
                        <p className="text-gray-400 text-xs">1 byte/param</p>
                      </div>
                    </div>
                  </div>
                  <div className="bg-red-800/50 rounded p-3">
                    <h5 className="text-red-200 font-semibold text-sm mb-2">Training Memory (FP16):</h5>
                    <ul className="text-gray-300 text-xs space-y-1">
                      <li>• <span className="text-blue-400">Model weights:</span> 14GB</li>
                      <li>• <span className="text-green-400">Gradients:</span> 14GB</li>
                      <li>• <span className="text-yellow-400">Optimizer states:</span> 28GB</li>
                      <li>• <span className="text-red-400">Total:</span> ~60GB minimum</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800/50 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-white mb-4 text-center">⚖️ Practical Decision Framework</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-blue-800/30 rounded p-4">
              <h4 className="text-blue-200 font-semibold text-lg mb-3 text-center">Model Selection</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-blue-200 font-semibold">Small &amp; Well-trained</p>
                  <p className="text-gray-300">Qwen2.5-0.5B (1GB)</p>
                  <p className="text-green-400">✓ Fits any GPU</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-purple-200 font-semibold">Medium Sweet Spot</p>
                  <p className="text-gray-300">Llama-3.2-1B (2GB)</p>
                  <p className="text-yellow-400">△ Consumer friendly</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-orange-200 font-semibold">Large Models</p>
                  <p className="text-gray-300">7B+ (14GB+)</p>
                  <p className="text-red-400">✗ Needs enterprise GPU</p>
                </div>
              </div>
            </div>
            
            <div className="bg-green-800/30 rounded p-4">
              <h4 className="text-green-200 font-semibold text-lg mb-3 text-center">GPU Requirements</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-green-200 font-semibold">RTX 4090 (24GB)</p>
                  <p className="text-gray-300">• 0.5-1B: Training ✓</p>
                  <p className="text-gray-300">• 7B: Inference only</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-blue-200 font-semibold">A100 (80GB)</p>
                  <p className="text-gray-300">• 7B: Training ✓</p>
                  <p className="text-gray-300">• 13B: Inference only</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-purple-200 font-semibold">H100 (80GB)</p>
                  <p className="text-gray-300">• 13B: Training ✓</p>
                  <p className="text-gray-300">• 30B+: Multi-GPU</p>
                </div>
              </div>
            </div>
            
            <div className="bg-amber-800/30 rounded p-4">
              <h4 className="text-amber-200 font-semibold text-lg mb-3 text-center">Training Strategy</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-amber-200 font-semibold">Chinchilla Optimal</p>
                  <p className="text-gray-300">20 tokens per parameter</p>
                  <p className="text-green-400">Best performance/compute</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-yellow-200 font-semibold">LoRA Fine-tuning</p>
                  <p className="text-gray-300">0.1% parameters</p>
                  <p className="text-blue-400">128× memory reduction</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-red-200 font-semibold">Quantization</p>
                  <p className="text-gray-300">INT8/4 precision</p>
                  <p className="text-purple-400">2-4× memory savings</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-4">
          <h4 className="text-amber-300 font-semibold mb-2">💡 Engineering Insight</h4>
          <p className="text-gray-300 text-sm">
            <strong>Combine scaling laws with memory constraints for smart decisions.</strong>
            Chinchilla taught us that data efficiency beats raw model size. 
            Understanding tensor shapes helps you debug OOM errors and optimize batch sizes for your hardware.
          </p>
        </div>
      </div>
    )
  },
  {
    title: "Modern Transformer Components",
    subtitle: "Evolution from Original 2017 Architecture",
    presenterNotes: [
      "Original 2017 transformer was just the beginning - modern LLMs use heavily optimized variants",
      "Each component here solved specific problems: training instability, memory usage, computation speed",
      "RMSNorm: Simpler normalization, GQA: Memory efficiency, RoPE: Better position encoding",
      "SwiGLU: Better activation function, Flash Attention: Memory optimization",
      "Understanding these helps you choose the right model architecture and debug training issues",
      "These aren't academic curiosities - they're in GPT-4, Claude, Llama, every production model"
    ],
    content: (
      <div className="space-y-6">
        <div className="bg-blue-900/20 border border-blue-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-blue-300 mb-4 text-center">🧠 Attention Mechanisms: Beyond Basic MHA</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-blue-200 font-semibold text-lg">Multi-Head Attention (MHA) - Original</h4>
              <div className="bg-blue-800/30 rounded p-4">
                <div className="bg-gray-900 rounded p-3 mb-3">
                  <code className="text-green-300 text-sm font-mono block">
                    heads = 8<br/>
                    each head: [batch, seq, d_head]<br/>
                    memory: O(seq² × heads)
                  </code>
                </div>
                <ul className="text-gray-300 text-xs space-y-1">
                  <li>• <span className="text-blue-400">Parallel attention heads</span> - each learns different patterns</li>
                  <li>• <span className="text-green-400">Proven architecture</span> - works reliably</li>
                  <li>• <span className="text-red-400">Memory hungry</span> - scales quadratically</li>
                  <li>• <span className="text-yellow-400">Full KV cache</span> - every head has full memory</li>
                </ul>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-purple-200 font-semibold text-lg">Grouped-Query Attention (GQA) - Modern</h4>
              <div className="bg-purple-800/30 rounded p-4">
                <div className="bg-gray-900 rounded p-3 mb-3">
                  <code className="text-green-300 text-sm font-mono block">
                    query_heads = 32<br/>
                    kv_heads = 8  # 4x fewer!<br/>
                    memory: 4x reduction
                  </code>
                </div>
                <div className="space-y-2">
                  <div className="bg-purple-900/50 rounded p-2">
                    <p className="text-purple-200 text-xs font-semibold">Evolution: MHA → MQA → GQA</p>
                    <p className="text-gray-300 text-xs">Balance between memory and performance</p>
                  </div>
                  <ul className="text-gray-300 text-xs space-y-1">
                    <li>• <span className="text-green-400">Memory efficient</span> - fewer KV heads</li>
                    <li>• <span className="text-blue-400">Performance maintained</span> - minimal quality loss</li>
                    <li>• <span className="text-purple-400">Used in Llama 2/3</span> - production proven</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-green-900/20 border border-green-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-green-300 mb-4 text-center">📍 Position Encodings: From Sin/Cos to RoPE</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-red-200 font-semibold text-lg">Original Sinusoidal (2017)</h4>
              <div className="bg-red-800/30 rounded p-4">
                <div className="bg-gray-900 rounded p-3 mb-3">
                  <code className="text-red-300 text-sm font-mono block">
                    PE(pos,2i) = sin(pos/10000^(2i/d))<br/>
                    PE(pos,2i+1) = cos(pos/10000^(2i/d))
                  </code>
                </div>
                <ul className="text-gray-300 text-xs space-y-1">
                  <li>• <span className="text-red-400">Added to embeddings</span> - separate from attention</li>
                  <li>• <span className="text-yellow-400">Fixed patterns</span> - doesn't adapt</li>
                  <li>• <span className="text-blue-400">Extrapolation problems</span> - fails on longer sequences</li>
                  <li>• <span className="text-gray-400">Still used in BERT-style models</span></li>
                </ul>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-green-200 font-semibold text-lg">RoPE (Rotary Position Embedding)</h4>
              <div className="bg-green-800/30 rounded p-4">
                <div className="bg-gray-900 rounded p-3 mb-3">
                  <code className="text-green-300 text-sm font-mono block">
                    # Applied during attention computation<br/>
                    Q_pos = rotate(Q, position)<br/>
                    K_pos = rotate(K, position)
                  </code>
                </div>
                <div className="space-y-2">
                  <div className="bg-green-900/50 rounded p-2">
                    <p className="text-green-200 text-xs font-semibold">🎯 Key Innovation: Position-aware attention</p>
                    <p className="text-gray-300 text-xs">Relative positions naturally emerge</p>
                  </div>
                  <ul className="text-gray-300 text-xs space-y-1">
                    <li>• <span className="text-green-400">Better extrapolation</span> - works on longer sequences</li>
                    <li>• <span className="text-blue-400">Integrated attention</span> - position in attention computation</li>
                    <li>• <span className="text-purple-400">Used in GPT-4, Llama</span> - modern standard</li>
                    <li>• <span className="text-yellow-400">Relative positions</span> - learns patterns between tokens</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-purple-900/20 border border-purple-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-purple-300 mb-4 text-center">⚡ Optimizations: Normalization & Activation</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-blue-200 font-semibold text-lg">RMSNorm vs LayerNorm</h4>
              <div className="bg-blue-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-red-800/30 rounded p-2">
                    <p className="text-red-200 text-xs font-semibold mb-1">LayerNorm (Original):</p>
                    <code className="text-red-300 text-xs font-mono block">
                      mean = x.mean()<br/>
                      std = x.std()<br/>
                      norm = (x - mean) / std
                    </code>
                  </div>
                  <div className="bg-green-800/30 rounded p-2">
                    <p className="text-green-200 text-xs font-semibold mb-1">RMSNorm (Modern):</p>
                    <code className="text-green-300 text-xs font-mono block">
                      rms = sqrt(x².mean())<br/>
                      norm = x / rms
                    </code>
                  </div>
                  <ul className="text-gray-300 text-xs space-y-1 mt-2">
                    <li>• <span className="text-green-400">20% faster</span> - no mean computation</li>
                    <li>• <span className="text-blue-400">Same performance</span> - simpler is better</li>
                    <li>• <span className="text-purple-400">Used in Llama, PaLM</span> - industry standard</li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-orange-200 font-semibold text-lg">SwiGLU Activation</h4>
              <div className="bg-orange-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-gray-900 rounded p-3">
                    <p className="text-orange-200 text-xs font-semibold mb-2">Gated Linear Unit Pattern:</p>
                    <code className="text-orange-300 text-xs font-mono block">
                      gate = W_gate(x)<br/>
                      up = W_up(x)<br/>
                      output = swish(gate) * up
                    </code>
                  </div>
                  <div className="bg-orange-900/50 rounded p-2">
                    <p className="text-orange-200 text-xs font-semibold">Why Gating Works:</p>
                    <p className="text-gray-300 text-xs">Controls information flow like a smart switch</p>
                  </div>
                  <ul className="text-gray-300 text-xs space-y-1">
                    <li>• <span className="text-green-400">Better quality</span> - vs standard ReLU/GELU</li>
                    <li>• <span className="text-blue-400">Gated mechanism</span> - selective information flow</li>
                    <li>• <span className="text-purple-400">Used in PaLM, LLaMA</span> - proven improvement</li>
                    <li>• <span className="text-yellow-400">Slight compute increase</span> - worth the quality gain</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800/50 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-white mb-4 text-center">🚀 Memory Optimization: Flash Attention</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-3">
              <h4 className="text-red-200 font-semibold text-lg">Standard Attention Memory Problem</h4>
              <div className="bg-red-800/30 rounded p-4">
                <div className="bg-gray-900 rounded p-3 mb-3">
                  <code className="text-red-300 text-sm font-mono">
                    # Naive attention<br/>
                    QK = Q @ K.T  # [batch, seq, seq]<br/>
                    scores = softmax(QK)<br/>
                    output = scores @ V
                  </code>
                </div>
                <ul className="text-gray-300 text-xs space-y-1">
                  <li>• <span className="text-red-400">O(seq²) memory</span> - stores full attention matrix</li>
                  <li>• <span className="text-yellow-400">GPU memory bottleneck</span> - limits sequence length</li>
                  <li>• <span className="text-blue-400">Multiple reads/writes</span> - inefficient GPU usage</li>
                </ul>
              </div>
            </div>
            
            <div className="space-y-3">
              <h4 className="text-green-200 font-semibold text-lg">Flash Attention Solution</h4>
              <div className="bg-green-800/30 rounded p-4">
                <div className="bg-gray-900 rounded p-3 mb-3">
                  <code className="text-green-300 text-sm font-mono">
                    # Tiled computation<br/>
                    for block in tiles:<br/>
                    &nbsp;&nbsp;compute_attention_block()<br/>
                    # Never stores full matrix
                  </code>
                </div>
                <div className="space-y-2">
                  <div className="bg-green-900/50 rounded p-2">
                    <p className="text-green-200 text-xs font-semibold">🎯 Key Innovation: Block-wise computation</p>
                    <p className="text-gray-300 text-xs">Recompute instead of store</p>
                  </div>
                  <ul className="text-gray-300 text-xs space-y-1">
                    <li>• <span className="text-green-400">4x longer sequences</span> - same memory</li>
                    <li>• <span className="text-blue-400">2x faster training</span> - better GPU utilization</li>
                    <li>• <span className="text-purple-400">Mathematically identical</span> - no approximation</li>
                    <li>• <span className="text-yellow-400">Enabled in PyTorch 2.0+</span> - just use it!</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-amber-800/30 rounded-xl p-4">
          <h3 className="text-lg font-semibold text-amber-300 mb-3 text-center">🔄 Evolution Timeline: 2017 → 2024</h3>
          <div className="grid grid-cols-4 gap-3 text-center">
            <div className="bg-gray-800 rounded p-3">
              <h4 className="text-blue-300 text-sm font-semibold">2017: Original</h4>
              <ul className="text-xs text-gray-400 mt-1 space-y-1">
                <li>• Sinusoidal PE</li>
                <li>• LayerNorm</li>
                <li>• ReLU/GELU</li>
                <li>• Standard MHA</li>
              </ul>
            </div>
            <div className="bg-gray-800 rounded p-3">
              <h4 className="text-green-300 text-sm font-semibold">2019-2020</h4>
              <ul className="text-xs text-gray-400 mt-1 space-y-1">
                <li>• RoPE introduced</li>
                <li>• GLU variants</li>
                <li>• RMSNorm adoption</li>
              </ul>
            </div>
            <div className="bg-gray-800 rounded p-3">
              <h4 className="text-purple-300 text-sm font-semibold">2021-2022</h4>
              <ul className="text-xs text-gray-400 mt-1 space-y-1">
                <li>• Flash Attention</li>
                <li>• MQA/GQA</li>
                <li>• SwiGLU</li>
              </ul>
            </div>
            <div className="bg-gray-800 rounded p-3">
              <h4 className="text-amber-300 text-sm font-semibold">2023-2024</h4>
              <ul className="text-xs text-gray-400 mt-1 space-y-1">
                <li>• Production adoption</li>
                <li>• GPT-4, Llama 3</li>
                <li>• Standard practice</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-4">
          <h4 className="text-amber-300 font-semibold mb-2">💡 Engineering Insight</h4>
          <p className="text-gray-300 text-sm">
            <strong>Modern transformers are highly optimized versions of the 2017 original.</strong>
            Each component solves specific engineering problems: memory usage (GQA, Flash Attention), 
            training stability (RMSNorm), sequence length (RoPE), and quality (SwiGLU). 
            Understanding these helps you choose models and debug training issues.
          </p>
        </div>
      </div>
    )
  },
  {
    title: "Quantization: The Memory Game Changer",
    subtitle: "From FP32 to 1-bit - Pushing the Limits",
    presenterNotes: [
      "Quantization is what makes LLMs accessible to everyone, not just BigTech",
      "Think of it like image compression - lossy but visually indistinguishable",
      "INT8: The sweet spot for most applications, minimal quality loss",
      "QLoRA: Revolutionary 4-bit quantization that enabled fine-tuning on consumer GPUs", 
      "1-bit models: Cutting-edge research, potentially game-changing",
      "Understanding quantization helps you choose the right precision for your use case and hardware"
    ],
    content: (
      <div className="space-y-6">
        <div className="bg-blue-900/20 border border-blue-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-blue-300 mb-4 text-center">🎯 The Precision Spectrum: Trading Bits for Memory</h3>
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-red-900/30 rounded-xl p-4 border border-red-600/30 text-center">
              <h4 className="text-red-300 font-bold text-lg mb-2">FP32</h4>
              <div className="space-y-2">
                <div className="bg-red-800/50 rounded p-2">
                  <p className="text-red-200 text-xl font-bold">4 bytes</p>
                  <p className="text-gray-400 text-xs">per parameter</p>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="bg-gray-800 rounded p-1">
                    <p className="text-red-200 font-semibold">7B Model:</p>
                    <p className="text-gray-300">28GB memory</p>
                  </div>
                  <div className="bg-red-800/30 rounded p-1">
                    <p className="text-red-200">Research only</p>
                    <p className="text-gray-400">Wasteful precision</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-yellow-900/30 rounded-xl p-4 border border-yellow-600/30 text-center">
              <h4 className="text-yellow-300 font-bold text-lg mb-2">FP16</h4>
              <div className="space-y-2">
                <div className="bg-yellow-800/50 rounded p-2">
                  <p className="text-yellow-200 text-xl font-bold">2 bytes</p>
                  <p className="text-gray-400 text-xs">per parameter</p>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="bg-gray-800 rounded p-1">
                    <p className="text-yellow-200 font-semibold">7B Model:</p>
                    <p className="text-gray-300">14GB memory</p>
                  </div>
                  <div className="bg-yellow-800/30 rounded p-1">
                    <p className="text-yellow-200">Training standard</p>
                    <p className="text-gray-400">Good quality/speed</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-green-900/30 rounded-xl p-4 border border-green-600/30 text-center">
              <h4 className="text-green-300 font-bold text-lg mb-2">INT8</h4>
              <div className="space-y-2">
                <div className="bg-green-800/50 rounded p-2">
                  <p className="text-green-200 text-xl font-bold">1 byte</p>
                  <p className="text-gray-400 text-xs">per parameter</p>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="bg-gray-800 rounded p-1">
                    <p className="text-green-200 font-semibold">7B Model:</p>
                    <p className="text-gray-300">7GB memory</p>
                  </div>
                  <div className="bg-green-800/30 rounded p-1">
                    <p className="text-green-200">Inference sweet spot</p>
                    <p className="text-gray-400">&lt;1% quality loss</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-purple-900/30 rounded-xl p-4 border border-purple-600/30 text-center">
              <h4 className="text-purple-300 font-bold text-lg mb-2">4-bit</h4>
              <div className="space-y-2">
                <div className="bg-purple-800/50 rounded p-2">
                  <p className="text-purple-200 text-xl font-bold">0.5 bytes</p>
                  <p className="text-gray-400 text-xs">per parameter</p>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="bg-gray-800 rounded p-1">
                    <p className="text-purple-200 font-semibold">7B Model:</p>
                    <p className="text-gray-300">3.5GB memory</p>
                  </div>
                  <div className="bg-purple-800/30 rounded p-1">
                    <p className="text-purple-200">QLoRA magic</p>
                    <p className="text-gray-400">Consumer GPU training</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-green-900/20 border border-green-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-green-300 mb-4 text-center">🔬 Quantization Techniques: How the Magic Works</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-blue-200 font-semibold text-lg">INT8 Quantization (LLM.int8)</h4>
              <div className="bg-blue-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-gray-900 rounded p-3">
                    <h5 className="text-blue-200 font-semibold text-sm mb-2">Linear Quantization:</h5>
                    <code className="text-green-300 text-xs font-mono block">
                      scale = max_val / 127<br/>
                      quantized = round(fp16_weight / scale)<br/>
                      int8_val = clamp(quantized, -128, 127)
                    </code>
                  </div>
                  <div className="bg-blue-900/50 rounded p-3">
                    <h5 className="text-blue-200 font-semibold text-sm mb-2">Mixed Precision Strategy:</h5>
                    <ul className="text-gray-300 text-xs space-y-1">
                      <li>• <span className="text-green-400">Outlier detection</span> - keep critical weights in FP16</li>
                      <li>• <span className="text-blue-400">Per-channel scaling</span> - different scales per output channel</li>
                      <li>• <span className="text-purple-400">Dynamic dequantization</span> - compute in FP16</li>
                    </ul>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center text-xs mt-3">
                    <div className="bg-green-800/30 rounded p-1">
                      <p className="text-green-300 font-bold">2×</p>
                      <p className="text-gray-400">Memory ↓</p>
                    </div>
                    <div className="bg-yellow-800/30 rounded p-1">
                      <p className="text-yellow-300 font-bold">15%</p>
                      <p className="text-gray-400">Slower</p>
                    </div>
                    <div className="bg-blue-800/30 rounded p-1">
                      <p className="text-blue-300 font-bold">&lt;1%</p>
                      <p className="text-gray-400">Quality loss</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-purple-200 font-semibold text-lg">QLoRA: 4-bit Revolution</h4>
              <div className="bg-purple-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-gray-900 rounded p-3">
                    <h5 className="text-purple-200 font-semibold text-sm mb-2">NormalFloat4 (NF4):</h5>
                    <code className="text-green-300 text-xs font-mono block">
                      # Optimized for normal distribution<br/>
                      nf4_bins = [-1, -0.69, -0.52, ..., 1]<br/>
                      quantized = find_nearest_bin(weight)
                    </code>
                  </div>
                  <div className="bg-purple-900/50 rounded p-3">
                    <h5 className="text-purple-200 font-semibold text-sm mb-2">Double Quantization:</h5>
                    <ul className="text-gray-300 text-xs space-y-1">
                      <li>• <span className="text-green-400">Quantize weights</span> - 4-bit storage</li>
                      <li>• <span className="text-blue-400">Quantize scaling factors</span> - 8-bit compression</li>
                      <li>• <span className="text-purple-400">LoRA gradients in FP16</span> - training precision</li>
                    </ul>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center text-xs mt-3">
                    <div className="bg-green-800/30 rounded p-1">
                      <p className="text-green-300 font-bold">4×</p>
                      <p className="text-gray-400">Memory ↓</p>
                    </div>
                    <div className="bg-yellow-800/30 rounded p-1">
                      <p className="text-yellow-300 font-bold">~20%</p>
                      <p className="text-gray-400">Slower</p>
                    </div>
                    <div className="bg-purple-800/30 rounded p-1">
                      <p className="text-purple-300 font-bold">~2%</p>
                      <p className="text-gray-400">Quality loss</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-amber-300 mb-4 text-center">🚀 Cutting Edge: 1-bit Models</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-red-200 font-semibold text-lg">The Challenge</h4>
              <div className="bg-red-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-gray-900 rounded p-3">
                    <h5 className="text-red-200 font-semibold text-sm mb-2">Extreme Quantization:</h5>
                    <code className="text-red-300 text-xs font-mono block">
                      # Only +1 or -1<br/>
                      weight = sign(original_weight)<br/>
                      # 32× memory reduction!
                    </code>
                  </div>
                  <ul className="text-gray-300 text-xs space-y-1">
                    <li>• <span className="text-red-400">Information loss</span> - massive precision reduction</li>
                    <li>• <span className="text-yellow-400">Training instability</span> - gradient flow issues</li>
                    <li>• <span className="text-blue-400">Quality degradation</span> - traditionally unusable</li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-green-200 font-semibold text-lg">BitNet: The Breakthrough</h4>
              <div className="bg-green-800/30 rounded p-4">
                <div className="space-y-3">
                  <div className="bg-gray-900 rounded p-3">
                    <h5 className="text-green-200 font-semibold text-sm mb-2">Training from Scratch:</h5>
                    <ul className="text-gray-300 text-xs space-y-1">
                      <li>• <span className="text-green-400">Custom architecture</span> - designed for 1-bit</li>
                      <li>• <span className="text-blue-400">Specialized training</span> - not post-training quantization</li>
                      <li>• <span className="text-purple-400">Competitive quality</span> - matches FP16 models</li>
                    </ul>
                  </div>
                  <div className="bg-green-900/50 rounded p-3">
                    <h5 className="text-green-200 font-semibold text-sm mb-2">Potential Impact:</h5>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="text-center">
                        <p className="text-green-300 font-bold text-lg">32×</p>
                        <p className="text-gray-400">Memory reduction</p>
                      </div>
                      <div className="text-center">
                        <p className="text-green-300 font-bold text-lg">10×</p>
                        <p className="text-gray-400">Speed increase</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800/50 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-white mb-4 text-center">🎯 Practical Decision Matrix</h3>
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-blue-800/30 rounded p-4">
              <h4 className="text-blue-200 font-semibold text-lg mb-3 text-center">Use Case</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-blue-200 font-semibold">Research/Training</p>
                  <p className="text-gray-300">Maximum quality needed</p>
                  <p className="text-green-400">→ FP16</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-purple-200 font-semibold">Production Inference</p>
                  <p className="text-gray-300">Balance speed/quality</p>
                  <p className="text-green-400">→ INT8</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-orange-200 font-semibold">Consumer Training</p>
                  <p className="text-gray-300">Limited GPU memory</p>
                  <p className="text-green-400">→ QLoRA</p>
                </div>
              </div>
            </div>
            
            <div className="bg-green-800/30 rounded p-4">
              <h4 className="text-green-200 font-semibold text-lg mb-3 text-center">Hardware Fit</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-green-200 font-semibold">RTX 4090 (24GB)</p>
                  <p className="text-gray-300">• 7B FP16: Inference only</p>
                  <p className="text-gray-300">• 7B INT8: ✓ Training</p>
                  <p className="text-gray-300">• 13B QLoRA: ✓ Training</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-blue-200 font-semibold">RTX 3080 (10GB)</p>
                  <p className="text-gray-300">• 7B QLoRA: ✓ Training</p>
                  <p className="text-gray-300">• 3B INT8: ✓ Training</p>
                </div>
              </div>
            </div>
            
            <div className="bg-purple-800/30 rounded p-4">
              <h4 className="text-purple-200 font-semibold text-lg mb-3 text-center">Quality Impact</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-purple-200 font-semibold">Critical Applications</p>
                  <p className="text-gray-300">Medical, Legal, Financial</p>
                  <p className="text-red-400">→ FP16/INT8 only</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-yellow-200 font-semibold">General Use</p>
                  <p className="text-gray-300">Chatbots, Content</p>
                  <p className="text-green-400">→ INT8/4-bit OK</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-orange-200 font-semibold">Experimental</p>
                  <p className="text-gray-300">Research, Prototyping</p>
                  <p className="text-green-400">→ Any precision</p>
                </div>
              </div>
            </div>
            
            <div className="bg-amber-800/30 rounded p-4">
              <h4 className="text-amber-200 font-semibold text-lg mb-3 text-center">Tools & Libraries</h4>
              <div className="space-y-2 text-xs">
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-amber-200 font-semibold">bitsandbytes</p>
                  <p className="text-gray-300">INT8, QLoRA support</p>
                  <p className="text-green-400">pip install bitsandbytes</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-blue-200 font-semibold">Transformers</p>
                  <p className="text-gray-300">Built-in quantization</p>
                  <p className="text-green-400">load_in_8bit=True</p>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <p className="text-purple-200 font-semibold">PEFT</p>
                  <p className="text-gray-300">QLoRA integration</p>
                  <p className="text-green-400">LoraConfig with 4-bit</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-4">
          <h4 className="text-amber-300 font-semibold mb-2">💡 Engineering Insight</h4>
          <p className="text-gray-300 text-sm">
            <strong>Quantization is the great democratizer of LLMs.</strong>
            It's what made fine-tuning possible on consumer hardware and enabled deployment at scale.
            Choose precision based on your constraints: memory (4-bit), speed (INT8), or quality (FP16).
            The future is 1-bit models trained from scratch - potentially 32× more efficient than today.
          </p>
        </div>
      </div>
    )
  },
  {
    title: "LoRA: The Game Changer",
    subtitle: "Low-Rank Adaptation for Efficient Fine-tuning",
    presenterNotes: [
      "This is what makes custom LLMs accessible to everyone, not just Google/OpenAI",
      "Traditional fine-tuning: Update all 7B parameters = 28GB GPU memory needed",
      "LoRA: Update only 0.1% of parameters = fits on consumer GPU with 4GB",
      "Key insight: Most model updates are low-rank (like Git diff vs full codebase)",
      "Think of it as monkey-patching the model instead of recompiling everything",
      "Production impact: You can fine-tune on your laptop what used to need a cluster"
    ],
    content: (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-red-900/20 border border-red-600/30 rounded-xl p-5">
            <h3 className="text-lg font-semibold text-red-300 mb-4">❌ Traditional Fine-tuning</h3>
            <div className="space-y-3">
              <div className="bg-red-800/30 rounded p-3">
                <h4 className="text-red-200 font-semibold text-sm">Update ALL Parameters</h4>
                <code className="block text-xs text-red-200 mt-2 font-mono">
                  7B params × 4 bytes = 28GB<br/>
                  + optimizer states = 84GB<br/>
                  + gradients = 112GB
                </code>
              </div>
              <div className="bg-red-800/30 rounded p-3">
                <h4 className="text-red-200 font-semibold text-sm">Hardware Requirements</h4>
                <ul className="text-gray-300 text-xs mt-2 space-y-1">
                  <li>• A100 80GB GPU ($10K+)</li>
                  <li>• Multi-GPU setup</li>
                  <li>• Cloud costs: $100s/hour</li>
                </ul>
              </div>
              <div className="text-center p-3 bg-red-800/50 rounded">
                <p className="text-red-200 font-bold">Accessible only to BigTech</p>
              </div>
            </div>
          </div>
          
          <div className="bg-green-900/20 border border-green-600/30 rounded-xl p-5">
            <h3 className="text-lg font-semibold text-green-300 mb-4">✅ LoRA Fine-tuning</h3>
            <div className="space-y-3">
              <div className="bg-green-800/30 rounded p-3">
                <h4 className="text-green-200 font-semibold text-sm">Update 0.1% Parameters</h4>
                <code className="block text-xs text-green-200 mt-2 font-mono">
                  7M trainable × 4 bytes = 28MB<br/>
                  + optimizer states = 84MB<br/>
                  + gradients = 112MB
                </code>
              </div>
              <div className="bg-green-800/30 rounded p-3">
                <h4 className="text-green-200 font-semibold text-sm">Hardware Requirements</h4>
                <ul className="text-gray-300 text-xs mt-2 space-y-1">
                  <li>• RTX 4090 24GB ($1.5K)</li>
                  <li>• Even RTX 3080 works</li>
                  <li>• Cloud costs: $2-5/hour</li>
                </ul>
              </div>
              <div className="text-center p-3 bg-green-800/50 rounded">
                <p className="text-green-200 font-bold">Accessible to everyone</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-6 border border-blue-600/30">
          <h3 className="text-xl font-semibold text-white mb-4 text-center">How LoRA Works: Matrix Factorization</h3>
          <div className="grid grid-cols-3 gap-6 items-center">
            <div className="text-center">
              <div className="bg-gray-800/50 rounded p-4 mb-2">
                <h4 className="text-white font-semibold mb-2">Original Weight</h4>
                <div className="bg-blue-600 h-16 w-16 mx-auto rounded mb-2"></div>
                <code className="text-xs text-blue-300">W ∈ ℝ^(4096×4096)</code>
                <p className="text-xs text-gray-300 mt-1">16M parameters</p>
              </div>
            </div>
            <div className="text-center">
              <p className="text-2xl text-white mb-2">≈</p>
              <p className="text-sm text-gray-300">decomposes into</p>
            </div>
            <div className="text-center">
              <div className="bg-gray-800/50 rounded p-4 mb-2">
                <h4 className="text-white font-semibold mb-2">Low-Rank Matrices</h4>
                <div className="flex justify-center gap-2 mb-2">
                  <div className="bg-purple-600 h-16 w-4 rounded"></div>
                  <span className="text-white self-center">×</span>
                  <div className="bg-purple-600 h-4 w-16 rounded"></div>
                </div>
                <code className="text-xs text-purple-300">B×A ∈ ℝ^(4096×16)×ℝ^(16×4096)</code>
                <p className="text-xs text-gray-300 mt-1">131K parameters (128× smaller!)</p>
              </div>
            </div>
          </div>
          <div className="mt-4 bg-gray-900/50 rounded-lg p-4 text-center">
            <code className="text-green-300 text-lg font-mono">
              W' = W₀ + α(BA)    where r=16 ≪ 4096
            </code>
            <p className="text-gray-300 text-sm mt-2">Frozen base + trainable low-rank adaptation</p>
          </div>
        </div>
        
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-4 text-center">
            <h4 className="text-amber-300 font-semibold text-lg">💾 Memory</h4>
            <p className="text-4xl font-bold text-amber-200 mt-2">128×</p>
            <p className="text-gray-300 text-sm">reduction</p>
            <p className="text-xs text-gray-400 mt-2">28GB → 220MB</p>
          </div>
          <div className="bg-blue-900/20 border border-blue-600/30 rounded-xl p-4 text-center">
            <h4 className="text-blue-300 font-semibold text-lg">⚡ Speed</h4>
            <p className="text-4xl font-bold text-blue-200 mt-2">3×</p>
            <p className="text-gray-300 text-sm">faster</p>
            <p className="text-xs text-gray-400 mt-2">Less compute needed</p>
          </div>
          <div className="bg-green-900/20 border border-green-600/30 rounded-xl p-4 text-center">
            <h4 className="text-green-300 font-semibold text-lg">🎯 Quality</h4>
            <p className="text-4xl font-bold text-green-200 mt-2">98%</p>
            <p className="text-gray-300 text-sm">retained</p>
            <p className="text-xs text-gray-400 mt-2">Minimal performance loss</p>
          </div>
        </div>
        
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-4">
          <h4 className="text-amber-300 font-semibold mb-2">💡 SWE Analogy</h4>
          <p className="text-gray-300 text-sm">
            <strong>Traditional fine-tuning:</strong> Recompiling entire codebase to change one feature<br/>
            <strong>LoRA:</strong> Adding a plugin/middleware that intercepts and modifies behavior
          </p>
        </div>
      </div>
    )
  },
  {
    title: "GPU Essentials",
    subtitle: "Hardware for Training",
    presenterNotes: [
      "GPU choice determines what models you can train and how fast",
      "Memory bandwidth is often more important than pure compute power",
      "A100s dominate enterprise training, but consumer GPUs (RTX 4090) viable for fine-tuning",
      "Flash Attention is game-changing - enables much larger context windows",
      "Consider TPUs for large-scale training, but GPUs offer more ecosystem flexibility",
      "Memory hierarchy: HBM &gt; DDR4 &gt; Disk - keep data close to compute"
    ],
    content: (
      <div className="space-y-6">
        <div className="space-y-4">
          <h3 className="text-xl font-bold text-white text-center">GPU Comparison</h3>
          <table className="w-full text-xs bg-gray-800/50 rounded-xl overflow-hidden">
            <thead><tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left p-2">GPU</th><th className="text-left p-2">Architecture</th><th className="text-right p-2">CUDA</th><th className="text-right p-2">TFLOPS</th><th className="text-right p-2">BW</th><th className="text-right p-2">VRAM</th><th className="text-right p-2">Retail Cost</th><th className="text-right p-2">Use Case</th>
            </tr></thead>
            <tbody className="text-gray-300">
              <tr className="border-b border-gray-700/50">
                <td className="p-2 font-semibold text-purple-400">H200</td>
                <td className="p-2 text-purple-300">Hopper</td>
                <td className="text-right p-2">16.9k</td>
                <td className="text-right p-2">989</td>
                <td className="text-right p-2">4.8TB/s</td>
                <td className="text-right p-2">141GB</td>
                <td className="text-right p-2 text-red-400">$45k+</td>
                <td className="text-right p-2 text-purple-400">Next-Gen</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-2 font-semibold text-blue-400">H100</td>
                <td className="p-2 text-blue-300">Hopper</td>
                <td className="text-right p-2">16.9k</td>
                <td className="text-right p-2">989</td>
                <td className="text-right p-2">3.3TB/s</td>
                <td className="text-right p-2">80GB</td>
                <td className="text-right p-2 text-red-400">$30k</td>
                <td className="text-right p-2 text-blue-400">Enterprise</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-2 font-semibold text-green-400">A100</td>
                <td className="p-2 text-green-300">Ampere</td>
                <td className="text-right p-2">6.9k</td>
                <td className="text-right p-2">312</td>
                <td className="text-right p-2">1.6TB/s</td>
                <td className="text-right p-2">80GB</td>
                <td className="text-right p-2 text-red-400">$15k</td>
                <td className="text-right p-2 text-green-400">Workhorse</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-2">V100</td>
                <td className="p-2 text-gray-400">Volta</td>
                <td className="text-right p-2">5.1k</td>
                <td className="text-right p-2">125</td>
                <td className="text-right p-2">900GB/s</td>
                <td className="text-right p-2">32GB</td>
                <td className="text-right p-2 text-gray-400">$3k</td>
                <td className="text-right p-2 text-gray-400">Legacy</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-2 font-semibold text-yellow-400">RTX 5090</td>
                <td className="p-2 text-yellow-300">Blackwell</td>
                <td className="text-right p-2">21.8k</td>
                <td className="text-right p-2">125</td>
                <td className="text-right p-2">1.8TB/s</td>
                <td className="text-right p-2">32GB</td>
                <td className="text-right p-2 text-yellow-400">$2k</td>
                <td className="text-right p-2 text-yellow-400">Consumer</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-2">RTX Pro 6000</td>
                <td className="p-2 text-orange-300">Blackwell</td>
                <td className="text-right p-2">24.1k</td>
                <td className="text-right p-2">125</td>
                <td className="text-right p-2">1.8TB/s</td>
                <td className="text-right p-2">96GB</td>
                <td className="text-right p-2 text-orange-400">$8k</td>
                <td className="text-right p-2 text-orange-400">Professional</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-2">RTX 6000 Ada</td>
                <td className="p-2 text-cyan-300">Ada Lovelace</td>
                <td className="text-right p-2">18.2k</td>
                <td className="text-right p-2">91</td>
                <td className="text-right p-2">960GB/s</td>
                <td className="text-right p-2">48GB</td>
                <td className="text-right p-2 text-cyan-400">$7k</td>
                <td className="text-right p-2 text-cyan-400">Workstation</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-2">P100</td>
                <td className="p-2 text-gray-400">Pascal</td>
                <td className="text-right p-2">3.6k</td>
                <td className="text-right p-2">18.7</td>
                <td className="text-right p-2">732GB/s</td>
                <td className="text-right p-2">16GB</td>
                <td className="text-right p-2 text-gray-400">$1k</td>
                <td className="text-right p-2 text-gray-400">Legacy</td>
              </tr>
              <tr>
                <td className="p-2">T4</td>
                <td className="p-2 text-gray-400">Turing</td>
                <td className="text-right p-2">2.6k</td>
                <td className="text-right p-2">65</td>
                <td className="text-right p-2">320GB/s</td>
                <td className="text-right p-2">16GB</td>
                <td className="text-right p-2 text-gray-400">$2k</td>
                <td className="text-right p-2 text-gray-400">Cloud/Inference</td>
              </tr>
            </tbody>
          </table>
        </div>
        
        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-white">Memory Matters Most</h3>
            <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
              <h4 className="font-semibold text-blue-300 mb-2">Model Size Limits</h4>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• <span className="text-green-400">7B model</span>: ~14GB (FP16)</li>
                <li>• <span className="text-yellow-400">13B model</span>: ~26GB (FP16)</li>
                <li>• <span className="text-red-400">70B model</span>: ~140GB (FP16)</li>
              </ul>
            </div>
            
            <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
              <h4 className="font-semibold text-purple-300 mb-2">Bandwidth vs Compute</h4>
              <p className="text-sm text-gray-300">
                Training is <span className="text-yellow-400 font-semibold">memory-bound</span>, not compute-bound. 
                Moving data between GPU memory layers is the bottleneck.
              </p>
            </div>
          </div>
          
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-white">Cost vs Performance</h3>
            <div className="bg-yellow-900/30 border border-yellow-600/50 p-4 rounded-xl">
              <h4 className="font-semibold text-yellow-300 mb-2">Sweet Spots</h4>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• <span className="text-yellow-400">RTX 5090</span>: Best consumer value</li>
                <li>• <span className="text-orange-400">RTX Pro 6000</span>: Huge VRAM for price</li>
                <li>• <span className="text-green-400">A100</span>: Enterprise standard</li>
              </ul>
            </div>
            
            <div className="bg-red-900/30 border border-red-600/50 p-4 rounded-xl">
              <h4 className="font-semibold text-red-300 mb-2">Cost Reality</h4>
              <p className="text-sm text-gray-300">
                Enterprise GPUs cost <span className="text-red-400 font-semibold">10-15×</span> more than consumer cards
                but offer ECC memory, better cooling, and enterprise support.
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-green-300 mb-2">🚀 Flash Attention: The Game Changer</h4>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-green-400 font-semibold">2-4× faster</span><br/>
              <span className="text-gray-300">Attention computation</span>
            </div>
            <div>
              <span className="text-green-400 font-semibold">10-20× less memory</span><br/>
              <span className="text-gray-300">No attention matrix storage</span>
            </div>
            <div>
              <span className="text-green-400 font-semibold">Longer sequences</span><br/>
              <span className="text-gray-300">Context windows up to 1M+</span>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gray-800/50 p-3 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-2">CUDA Ecosystem</h4>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• Custom kernels for optimization</li>
              <li>• cuDNN for neural networks</li>
              <li>• Triton for kernel development</li>
            </ul>
          </div>
          
          <div className="bg-gray-800/50 p-3 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-2">Cost Considerations</h4>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• H200: $8-15/hour (cloud)</li>
              <li>• H100: $4-8/hour (cloud)</li>
              <li>• RTX 5090: $2500 (purchase)</li>
            </ul>
          </div>
          
          <div className="bg-gray-800/50 p-3 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-2">GPU Metrics Explained</h4>
            <div className="text-xs text-gray-400 space-y-1">
              <div><span className="text-blue-400 font-semibold">CUDA Cores:</span> Parallel processors for computation</div>
              <div><span className="text-green-400 font-semibold">TFLOPS:</span> Trillion floating-point operations/second</div>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Critical Hyperparameters",
    subtitle: "Ranked by Impact",
    presenterNotes: [
      "Learning rate is THE most critical parameter - can make or break training",
      "Too high = NaN loss explosion, too low = no learning progress",
      "Batch size affects both stability and memory usage - bigger is generally better",
      "Warmup prevents catastrophic early updates when model weights are random",
      "LR scaling rule: <1B=5e-4, 1-7B=1e-4, >7B=5e-5",
      "Weight decay prevents overfitting - acts as L2 regularization",
      "Gradient clipping is essential for transformer stability - prevents exploding gradients"
    ],
    content: (
      <div className="space-y-6">
        <div className="space-y-4">
          <h3 className="text-xl font-bold text-white text-center">Impact Hierarchy</h3>
          <table className="w-full text-sm bg-gray-800/50 rounded-xl overflow-hidden">
            <thead><tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left p-3">Parameter</th><th className="text-center p-3">Impact Level</th><th className="text-center p-3">Typical Range</th><th className="text-left p-3">Notes</th>
            </tr></thead>
            <tbody className="text-gray-300">
              <tr className="border-b border-gray-700/50">
                <td className="p-3 font-semibold text-red-400">Learning Rate</td>
                <td className="text-center p-3"><span className="bg-red-900/50 px-2 py-1 rounded text-red-300 font-semibold">Critical</span></td>
                <td className="text-center p-3">1e-5 to 1e-3</td>
                <td className="text-left p-3">Too high = divergence, too low = no learning</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-3 font-semibold">Batch Size</td>
                <td className="text-center p-3"><span className="bg-orange-900/50 px-2 py-1 rounded text-orange-300 font-semibold">High</span></td>
                <td className="text-center p-3">8 to 512</td>
                <td className="text-left p-3">Larger = more stable, but needs more memory</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-3 font-semibold">Warmup Steps</td>
                <td className="text-center p-3"><span className="bg-orange-900/50 px-2 py-1 rounded text-orange-300 font-semibold">High</span></td>
                <td className="text-center p-3">3-10% of total</td>
                <td className="text-left p-3">Prevents early training instability</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-3">Weight Decay</td>
                <td className="text-center p-3"><span className="bg-yellow-900/50 px-2 py-1 rounded text-yellow-300">Medium</span></td>
                <td className="text-center p-3">0.01 to 0.1</td>
                <td className="text-left p-3">L2 regularization to prevent overfitting</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-3">Gradient Clip</td>
                <td className="text-center p-3"><span className="bg-yellow-900/50 px-2 py-1 rounded text-yellow-300">Medium</span></td>
                <td className="text-center p-3">0.5 to 1.0</td>
                <td className="text-left p-3">Prevents exploding gradients</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-3">Dropout</td>
                <td className="text-center p-3"><span className="bg-gray-700/50 px-2 py-1 rounded text-gray-400">Low</span></td>
                <td className="text-center p-3">0.0 to 0.1</td>
                <td className="text-left p-3">Often disabled (0.0) for modern LLMs</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-3">Max Sequence Length</td>
                <td className="text-center p-3"><span className="bg-yellow-900/50 px-2 py-1 rounded text-yellow-300">Medium</span></td>
                <td className="text-center p-3">512 to 4096</td>
                <td className="text-left p-3">Memory scales quadratically with length</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-3">Mixed Precision</td>
                <td className="text-center p-3"><span className="bg-yellow-900/50 px-2 py-1 rounded text-yellow-300">Medium</span></td>
                <td className="text-center p-3">fp16/bf16</td>
                <td className="text-left p-3">2× memory savings, 2-3× speedup</td>
              </tr>
              <tr>
                <td className="p-3">Adam β₂</td>
                <td className="text-center p-3"><span className="bg-gray-700/50 px-2 py-1 rounded text-gray-400">Low</span></td>
                <td className="text-center p-3">0.95 to 0.999</td>
                <td className="text-left p-3">0.95 better for long sequences</td>
              </tr>
            </tbody>
          </table>
        </div>
        
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-red-900/30 border border-red-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-red-300 mb-2">Learning Rate Scaling</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• <span className="text-green-400">&lt;1B params</span>: 5e-4 to 1e-3</li>
              <li>• <span className="text-yellow-400">1-7B params</span>: 1e-4 to 5e-4</li>
              <li>• <span className="text-red-400">&gt;7B params</span>: 5e-5 to 2e-4</li>
            </ul>
          </div>
          
          <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-blue-300 mb-2">Cosine Schedule</h4>
            <p className="text-sm text-gray-300">
              <span className="text-blue-400 font-semibold">Warmup</span> → <span className="text-yellow-400 font-semibold">Peak LR</span> → <span className="text-green-400 font-semibold">Cosine Decay</span>
            </p>
            <p className="text-xs text-gray-400 mt-2">Smooth transitions prevent training instability</p>
          </div>
          
          <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-purple-300 mb-2">Effective Batch Size</h4>
            <p className="text-sm text-gray-300">
              <span className="text-purple-400 font-semibold">micro_batch × grad_accum × num_gpus</span>
            </p>
            <p className="text-xs text-gray-400 mt-2">Target: 128-1024 for stable training</p>
          </div>
        </div>
        
        <div className="bg-gray-800/50 p-3 rounded-xl">
          <h4 className="font-semibold text-gray-300 mb-2">🔧 Quick Setup Rules</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-yellow-400 font-semibold">Start Conservative:</span>
              <ul className="text-gray-300 text-xs mt-1 space-y-1">
                <li>• LR: 1e-4 for most models</li>
                <li>• Warmup: 5% of total steps</li>
                <li>• Sequence length: 512 (then increase)</li>
                <li>• Mixed precision: bf16 on modern GPUs</li>
              </ul>
            </div>
            <div>
              <span className="text-red-400 font-semibold">Common Failures:</span>
              <ul className="text-gray-300 text-xs mt-1 space-y-1">
                <li>• Loss = NaN → Reduce LR by 10×</li>
                <li>• OOM error → Reduce batch/sequence length</li>
                <li>• No learning → Increase LR, check data</li>
                <li>• Weight decay: 0.01</li>
              </ul>
            </div>
            <div>
              <span className="text-green-400 font-semibold">Monitor &amp; Adjust:</span>
              <ul className="text-gray-300 text-xs mt-1 space-y-1">
                <li>• Loss spikes = reduce LR</li>
                <li>• Slow progress = increase LR</li>
                <li>• NaN = restart with lower LR</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "⚠️ Model Selection: The Specialization Trap",
    subtitle: "Why CodeLlama Fails at New Languages",
    presenterNotes: [
      "THIS IS CRITICAL - wrong choice here wastes weeks of training time",
      "Real example: We tried fine-tuning CodeLlama on Rust, it kept generating Python",
      "Same model (base Llama) succeeded when we started from base instead of CodeLlama",
      "Think of specialized models like compiled C++ - optimized but inflexible",
      "Base models are like Python - slower but adaptable to any use case",
      "For workshop: We're using Qwen2.5-0.5B base because it's flexible and fast to train"
    ],
    content: (
      <div className="space-y-6">
        <div className="bg-red-900/20 border border-red-600/30 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-red-300 mb-4 text-center">🚨 The Trap: Why Specialized Models Fail</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-3">
              <h4 className="text-red-200 font-semibold text-lg">Real Examples That Failed:</h4>
              <div className="space-y-2">
                <div className="bg-red-800/30 rounded p-3">
                  <p className="text-red-200 font-semibold text-sm">CodeLlama → New Programming Language</p>
                  <p className="text-gray-300 text-xs">Kept generating Python syntax instead of learning new language</p>
                </div>
                <div className="bg-red-800/30 rounded p-3">
                  <p className="text-red-200 font-semibold text-sm">BioMedLM → Legal Documents</p>
                  <p className="text-gray-300 text-xs">Medical terminology interfered with legal concept learning</p>
                </div>
                <div className="bg-red-800/30 rounded p-3">
                  <p className="text-red-200 font-semibold text-sm">Llama-Instruct → Different Personality</p>
                  <p className="text-gray-300 text-xs">RLHF alignment resisted behavior changes</p>
                </div>
              </div>
            </div>
            <div className="space-y-3">
              <h4 className="text-green-200 font-semibold text-lg">What Worked Instead:</h4>
              <div className="space-y-2">
                <div className="bg-green-800/30 rounded p-3">
                  <p className="text-green-200 font-semibold text-sm">Base Llama → New Programming Language</p>
                  <p className="text-gray-300 text-xs">Successfully learned syntax and semantics</p>
                </div>
                <div className="bg-green-800/30 rounded p-3">
                  <p className="text-green-200 font-semibold text-sm">Base Qwen → Legal Documents</p>
                  <p className="text-gray-300 text-xs">Clean slate learned legal reasoning patterns</p>
                </div>
                <div className="bg-green-800/30 rounded p-3">
                  <p className="text-green-200 font-semibold text-sm">Base Model → Custom Personality</p>
                  <p className="text-gray-300 text-xs">Flexible enough to adopt new behaviors</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-800/50 rounded-xl p-5">
          <h3 className="text-xl font-semibold text-white mb-4 text-center">Model Rigidity Spectrum</h3>
          <div className="space-y-4">
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-green-900/30 rounded-xl p-4 border border-green-600/30 text-center">
                <h4 className="text-green-300 font-semibold text-lg">Base Models</h4>
                <div className="w-full bg-green-600 h-3 rounded mt-2 mb-2"></div>
                <p className="text-xs text-gray-300 mb-2">100% Flexible</p>
                <ul className="text-xs text-gray-300 space-y-1">
                  <li>• Qwen2.5-base</li>
                  <li>• Llama-3.2-base</li>
                  <li>• Phi-3-base</li>
                </ul>
                <p className="text-green-200 text-xs mt-2 font-semibold">✓ Use for novel patterns</p>
              </div>
              
              <div className="bg-yellow-900/30 rounded-xl p-4 border border-yellow-600/30 text-center">
                <h4 className="text-yellow-300 font-semibold text-lg">Domain Specialized</h4>
                <div className="w-3/4 bg-yellow-600 h-3 rounded mt-2 mb-2"></div>
                <p className="text-xs text-gray-300 mb-2">75% Flexible</p>
                <ul className="text-xs text-gray-300 space-y-1">
                  <li>• CodeLlama</li>
                  <li>• BioMedLM</li>
                  <li>• FinanceGPT</li>
                </ul>
                <p className="text-yellow-200 text-xs mt-2 font-semibold">△ Use to extend domain</p>
              </div>
              
              <div className="bg-orange-900/30 rounded-xl p-4 border border-orange-600/30 text-center">
                <h4 className="text-orange-300 font-semibold text-lg">Instruction Tuned</h4>
                <div className="w-1/2 bg-orange-600 h-3 rounded mt-2 mb-2"></div>
                <p className="text-xs text-gray-300 mb-2">50% Flexible</p>
                <ul className="text-xs text-gray-300 space-y-1">
                  <li>• Llama-Instruct</li>
                  <li>• Qwen-Chat</li>
                  <li>• Mistral-Instruct</li>
                </ul>
                <p className="text-orange-200 text-xs mt-2 font-semibold">⚠ Avoid for training</p>
              </div>
              
              <div className="bg-red-900/30 rounded-xl p-4 border border-red-600/30 text-center">
                <h4 className="text-red-300 font-semibold text-lg">RLHF Aligned</h4>
                <div className="w-1/4 bg-red-600 h-3 rounded mt-2 mb-2"></div>
                <p className="text-xs text-gray-300 mb-2">25% Flexible</p>
                <ul className="text-xs text-gray-300 space-y-1">
                  <li>• ChatGPT</li>
                  <li>• Claude</li>
                  <li>• GPT-4</li>
                </ul>
                <p className="text-red-200 text-xs mt-2 font-semibold">✗ Nearly impossible</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-blue-900/20 border border-blue-600/30 rounded-xl p-4">
            <h4 className="text-blue-300 font-semibold text-lg mb-3">🎯 Decision Framework</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-green-400 mt-1">✓</span>
                <span className="text-gray-300"><strong>Use Base Models when:</strong><br/>
                • Learning new patterns/languages<br/>
                • Cross-domain transfer<br/>
                • Behavioral changes</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-yellow-400 mt-1">△</span>
                <span className="text-gray-300"><strong>Use Specialized when:</strong><br/>
                • Extending within same domain<br/>
                • Limited training data<br/>
                • Want domain knowledge preserved</span>
              </div>
            </div>
          </div>
          
          <div className="bg-green-900/20 border border-green-600/30 rounded-xl p-4">
            <h4 className="text-green-300 font-semibold text-lg mb-3">🚀 Today's Recommendation</h4>
            <div className="space-y-3">
              <div className="bg-green-800/30 rounded p-3">
                <h5 className="text-green-200 font-semibold">Primary: Qwen2.5-0.5B</h5>
                <ul className="text-xs text-gray-300 mt-1 space-y-1">
                  <li>• 494M parameters (perfect for laptops)</li>
                  <li>• Base model (maximum flexibility)</li>
                  <li>• Fast training (minutes, not hours)</li>
                  <li>• Excellent for learning</li>
                </ul>
              </div>
              <div className="bg-blue-800/30 rounded p-3">
                <h5 className="text-blue-200 font-semibold">Alternative: Llama-3.2-1B</h5>
                <ul className="text-xs text-gray-300 mt-1 space-y-1">
                  <li>• 1B parameters (still manageable)</li>
                  <li>• Meta's latest architecture</li>
                  <li>• Strong performance/size ratio</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-amber-900/20 border border-amber-600/30 rounded-xl p-4">
          <h4 className="text-amber-300 font-semibold mb-2">💡 Engineering Insight</h4>
          <p className="text-gray-300 text-sm">
            <strong>Model specialization is like code compilation:</strong> Optimized for specific use cases but inflexible.
            Choose base models when you need adaptability, specialized when extending existing functionality.
          </p>
        </div>
      </div>
    )
  },
  {
    title: "MLOps Workflow",
    subtitle: "Complete Pipeline",
    presenterNotes: [
      "MLOps is DevOps for ML - reproducible, scalable training pipelines",
      "Configuration management is crucial - use Hydra for composable configs",
      "Track everything with Weights & Biases - metrics, artifacts, hyperparameters", 
      "Checkpoint frequently for fault tolerance and experiment resumption",
      "Data pipeline is often the bottleneck - optimize formats and loading",
      "Infrastructure matters: local prototyping → cloud training → production serving",
      "Open-source data lakes: MinIO for S3-compatible storage, lakeFS for versioning, Iceberg for tables"
    ],
    content: (
      <div className="space-y-6">
        <div className="space-y-4">
          <h3 className="text-xl font-bold text-white text-center">End-to-End Training Pipeline</h3>
          <div className="bg-gray-800/50 p-4 rounded-xl">
            <div className="flex items-center justify-center gap-2 text-sm flex-wrap">
              <span className="bg-blue-900/50 px-3 py-2 rounded-lg text-blue-200 font-semibold">📋 Config</span>
              <span className="text-gray-500 text-xl">→</span>
              <span className="bg-purple-900/50 px-3 py-2 rounded-lg text-purple-200 font-semibold">📊 Data</span>
              <span className="text-gray-500 text-xl">→</span>
              <span className="bg-green-900/50 px-3 py-2 rounded-lg text-green-200 font-semibold">🤖 Model</span>
              <span className="text-gray-500 text-xl">→</span>
              <span className="bg-orange-900/50 px-3 py-2 rounded-lg text-orange-200 font-semibold">🎯 LoRA</span>
              <span className="text-gray-500 text-xl">→</span>
              <span className="bg-red-900/50 px-3 py-2 rounded-lg text-red-200 font-semibold">🚀 Train</span>
              <span className="text-gray-500 text-xl">→</span>
              <span className="bg-yellow-900/50 px-3 py-2 rounded-lg text-yellow-200 font-semibold">✅ Eval</span>
              <span className="text-gray-500 text-xl">→</span>
              <span className="bg-cyan-900/50 px-3 py-2 rounded-lg text-cyan-200 font-semibold">💾 Deploy</span>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-4">
            <h4 className="text-lg font-bold text-white">🔧 Configuration Management</h4>
            <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
              <h5 className="font-semibold text-blue-300 mb-2">Hydra Config Structure</h5>
              <div className="bg-gray-800/50 p-3 rounded text-xs font-mono">
                <div className="text-gray-300">config/</div>
                <div className="text-gray-300 ml-2">├── config.yaml</div>
                <div className="text-gray-300 ml-2">├── model/</div>
                <div className="text-green-400 ml-4">│   ├── qwen_0.5b.yaml</div>
                <div className="text-green-400 ml-4">│   └── llama_1b.yaml</div>
                <div className="text-gray-300 ml-2">└── training/</div>
                <div className="text-blue-400 ml-4">    ├── quick.yaml</div>
                <div className="text-blue-400 ml-4">    └── production.yaml</div>
              </div>
              <div className="mt-2 text-xs text-gray-300">
                <code className="bg-gray-700 px-2 py-1 rounded">python train.py model=qwen_0.5b training=quick</code>
              </div>
            </div>
            
            <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
              <h5 className="font-semibold text-purple-300 mb-2">Data Pipeline Formats</h5>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• <span className="text-yellow-400">JSONL</span>: Simple, human-readable</li>
                <li>• <span className="text-green-400">Parquet</span>: Columnar, compressed, fast</li>
                <li>• <span className="text-blue-400">Arrow</span>: Memory-mapped, zero-copy</li>
                <li>• <span className="text-purple-400">HF Datasets</span>: Standardized API</li>
              </ul>
            </div>
          </div>
          
          <div className="space-y-4">
            <h4 className="text-lg font-bold text-white">📊 Experiment Tracking</h4>
            <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
              <h5 className="font-semibold text-green-300 mb-2">Weights &amp; Biases Integration</h5>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• <span className="text-green-400">Metrics</span>: Loss curves, perplexity</li>
                <li>• <span className="text-blue-400">System</span>: GPU utilization, memory</li>
                <li>• <span className="text-purple-400">Artifacts</span>: Model checkpoints</li>
                <li>• <span className="text-yellow-400">Hyperparams</span>: Automatic logging</li>
              </ul>
            </div>
            
            <div className="bg-orange-900/30 border border-orange-600/50 p-4 rounded-xl">
              <h5 className="font-semibold text-orange-300 mb-2">Infrastructure Layers</h5>
              <div className="space-y-2 text-sm text-gray-300">
                <div><span className="text-orange-400">Compute</span>: Local GPU → Cloud scaling</div>
                <div><span className="text-blue-400">Storage</span>: MinIO (local S3), lakeFS (versioning)</div>
                <div><span className="text-green-400">Orchestration</span>: K8s, Slurm, Ray</div>
                <div><span className="text-purple-400">Serving</span>: TGI, vLLM, Triton</div>
              </div>
            </div>
            
            <div className="bg-cyan-900/30 border border-cyan-600/50 p-4 rounded-xl">
              <h5 className="font-semibold text-cyan-300 mb-2">OSS Data Lakes</h5>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• <span className="text-cyan-400">MinIO</span>: S3-compatible local storage</li>
                <li>• <span className="text-green-400">lakeFS</span>: Git-like data versioning</li>
                <li>• <span className="text-blue-400">Apache Iceberg</span>: Table format leader</li>
                <li>• <span className="text-purple-400">Kubeflow</span>: Full MLOps platform</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gray-800/50 p-3 rounded-xl">
            <h5 className="font-semibold text-gray-300 mb-2">🔄 Checkpoint Strategy</h5>
            <div className="text-xs text-gray-400 space-y-1">
              <div>• Save every 500 steps</div>
              <div>• Keep last 3 checkpoints</div>
              <div>• Auto-resume on failure</div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 p-3 rounded-xl">
            <h5 className="font-semibold text-gray-300 mb-2">🔍 Quality Monitoring</h5>
            <div className="text-xs text-gray-400 space-y-1">
              <div>• Gradient norms tracking</div>
              <div>• Generation samples</div>
              <div>• Validation perplexity</div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 p-3 rounded-xl">
            <h5 className="font-semibold text-gray-300 mb-2">⚡ Optimization</h5>
            <div className="text-xs text-gray-400 space-y-1">
              <div>• Mixed precision (FP16)</div>
              <div>• Gradient accumulation</div>
              <div>• Efficient data loading</div>
            </div>
          </div>
        </div>
        
        <div className="bg-yellow-900/30 border border-yellow-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-yellow-300 mb-2">🎯 Production-Ready Checklist</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-yellow-400 font-semibold">Development:</span>
              <ul className="text-gray-300 text-xs mt-1 space-y-1">
                <li>✓ Reproducible configs</li>
                <li>✓ Version control integration</li>
                <li>✓ Automated data validation</li>
              </ul>
            </div>
            <div>
              <span className="text-green-400 font-semibold">Production:</span>
              <ul className="text-gray-300 text-xs mt-1 space-y-1">
                <li>✓ Model registry &amp; versioning</li>
                <li>✓ A/B testing framework</li>
                <li>✓ Performance monitoring</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Training Diagnostics",
    subtitle: "First-Glance Troubleshooting",
    presenterNotes: [
      "This slide is your emergency toolkit when training goes wrong",
      "Loss = NaN almost always means learning rate too high - reduce by 10x immediately",
      "Flat loss curve means learning rate too low OR bad/corrupted data",
      "GPU memory issues: use gradient accumulation to simulate larger batches",
      "Loss spikes indicate bad training samples - gradient clipping helps",
      "Monitor gradient norms: >100 is dangerous, <0.1 means no learning",
      "Always enable mixed precision (FP16) for 2-3x speedup on modern GPUs"
    ],
    content: (
      <div className="space-y-6">
        <div className="space-y-4">
          <h3 className="text-xl font-bold text-white text-center">Quick Diagnosis Guide</h3>
          <table className="w-full text-sm bg-gray-800/50 rounded-xl overflow-hidden">
            <thead><tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left p-3">🚨 Symptom</th><th className="text-left p-3">🔍 Likely Cause</th><th className="text-left p-3">🔧 Quick Fix</th><th className="text-left p-3">⏱️ Action</th>
            </tr></thead>
            <tbody className="text-gray-300">
              <tr className="border-b border-gray-700/50">
                <td className="p-3 font-semibold text-red-400">Loss = NaN</td>
                <td className="p-3">Learning rate too high</td>
                <td className="p-3 text-green-400">Reduce LR by 10x</td>
                <td className="p-3 text-yellow-400">Restart immediately</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-3 font-semibold text-orange-400">Flat loss curve</td>
                <td className="p-3">LR too low or bad data</td>
                <td className="p-3 text-green-400">Increase LR, check data quality</td>
                <td className="p-3 text-yellow-400">Adjust &amp; continue</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-3 font-semibold text-red-400">GPU OOM Error</td>
                <td className="p-3">Batch size too large</td>
                <td className="p-3 text-green-400">Use gradient accumulation</td>
                <td className="p-3 text-yellow-400">Restart with new config</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-3 font-semibold text-purple-400">Loss spikes</td>
                <td className="p-3">Bad data samples</td>
                <td className="p-3 text-green-400">Enable gradient clipping (1.0)</td>
                <td className="p-3 text-green-400">Continue training</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="p-3 font-semibold text-blue-400">Very slow training</td>
                <td className="p-3">No mixed precision</td>
                <td className="p-3 text-green-400">Enable FP16/BF16</td>
                <td className="p-3 text-green-400">2-3x speedup</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold text-yellow-400">Eval worse than train</td>
                <td className="p-3">Overfitting</td>
                <td className="p-3 text-green-400">Early stopping, more data</td>
                <td className="p-3 text-yellow-400">Stop &amp; analyze</td>
              </tr>
            </tbody>
          </table>
        </div>
        
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-red-900/30 border border-red-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-red-300 mb-2">🚨 Emergency Signals</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• <span className="text-red-400">Gradient norm &gt; 100</span>: Exploding</li>
              <li>• <span className="text-orange-400">Gradient norm &lt; 0.1</span>: Vanishing</li>
              <li>• <span className="text-yellow-400">GPU temp &gt; 85°C</span>: Thermal throttle</li>
              <li>• <span className="text-purple-400">Memory usage &gt; 95%</span>: Near OOM</li>
            </ul>
          </div>
          
          <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-green-300 mb-2">✅ Healthy Training</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• <span className="text-green-400">Smooth loss decrease</span></li>
              <li>• <span className="text-blue-400">Gradient norm 1-10</span></li>
              <li>• <span className="text-yellow-400">GPU utilization &gt; 90%</span></li>
              <li>• <span className="text-purple-400">Memory usage 80-90%</span></li>
            </ul>
          </div>
          
          <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-blue-300 mb-2">🔧 Essential Tools</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• <span className="text-blue-400">nvidia-smi</span>: GPU monitoring</li>
              <li>• <span className="text-green-400">wandb</span>: Real-time metrics</li>
              <li>• <span className="text-yellow-400">htop</span>: CPU/RAM usage</li>
              <li>• <span className="text-purple-400">torch.profiler</span>: Bottlenecks</li>
            </ul>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">📊 Key Metrics to Watch</h4>
            <div className="space-y-2 text-sm text-gray-300">
              <div className="flex justify-between">
                <span>Training Loss:</span>
                <span className="text-green-400">Steadily decreasing</span>
              </div>
              <div className="flex justify-between">
                <span>Validation Loss:</span>
                <span className="text-blue-400">Following train loss</span>
              </div>
              <div className="flex justify-between">
                <span>Learning Rate:</span>
                <span className="text-yellow-400">Following schedule</span>
              </div>
              <div className="flex justify-between">
                <span>GPU Memory:</span>
                <span className="text-purple-400">Consistent usage</span>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">⚡ Performance Optimization</h4>
            <div className="space-y-2 text-sm text-gray-300">
              <div className="flex justify-between">
                <span>Mixed Precision:</span>
                <span className="text-green-400">2-3x speedup</span>
              </div>
              <div className="flex justify-between">
                <span>Gradient Accumulation:</span>
                <span className="text-blue-400">Simulate larger batch</span>
              </div>
              <div className="flex justify-between">
                <span>Data Loading:</span>
                <span className="text-yellow-400">num_workers=4-8</span>
              </div>
              <div className="flex justify-between">
                <span>Compilation:</span>
                <span className="text-purple-400">torch.compile() +20%</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-yellow-900/30 border border-yellow-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-yellow-300 mb-2">🎯 Rule of Thumb: When to Restart vs Continue</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-red-400 font-semibold">🛑 Restart Training:</span>
              <ul className="text-gray-300 text-xs mt-1 space-y-1">
                <li>• Loss = NaN (learning rate too high)</li>
                <li>• No progress after 1000 steps</li>
                <li>• GPU OOM errors</li>
                <li>• Gradient norms consistently &gt; 100</li>
              </ul>
            </div>
            <div>
              <span className="text-green-400 font-semibold">▶️ Adjust and Continue:</span>
              <ul className="text-gray-300 text-xs mt-1 space-y-1">
                <li>• Occasional loss spikes (add grad clip)</li>
                <li>• Slow but steady progress</li>
                <li>• Minor performance issues</li>
                <li>• Validation slightly worse than train</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Memory Optimization",
    subtitle: "Fitting Large Models on Limited GPUs",
    presenterNotes: [
      "Memory is the biggest constraint in LLM training - not compute power",
      "Gradient checkpointing trades compute for memory - 30-50% memory reduction",
      "Mixed precision gives 2x memory savings with minimal quality impact",
      "QLoRA enables 7B model training on 16GB GPUs with <1% quality loss",
      "Flash Attention is game-changing - 10x memory reduction for attention",
      "Combine techniques: QLoRA + GradCheckpt + FlashAttn = train 7B on consumer GPUs"
    ],
    content: (
      <div className="space-y-6">
        <div className="space-y-4">
          <h3 className="text-xl font-bold text-white text-center">Memory-Efficient Training Stack</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
                <h4 className="font-semibold text-blue-300 mb-3">🔄 Gradient Checkpointing</h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <div className="flex justify-between">
                    <span>Memory Reduction:</span>
                    <span className="text-green-400 font-semibold">30-50%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Speed Impact:</span>
                    <span className="text-yellow-400">20-30% slower</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Mechanism:</span>
                    <span className="text-blue-400">Recompute activations</span>
                  </div>
                </div>
                <div className="mt-3 text-xs text-gray-400">
                  Trades compute for memory by discarding intermediate activations and recomputing during backward pass
                </div>
              </div>
              
              <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
                <h4 className="font-semibold text-purple-300 mb-3">🎯 Mixed Precision (FP16/BF16)</h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <div className="flex justify-between">
                    <span>Memory Savings:</span>
                    <span className="text-green-400 font-semibold">2x reduction</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Speed Boost:</span>
                    <span className="text-green-400 font-semibold">2-3x faster</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Quality Impact:</span>
                    <span className="text-green-400">Minimal with BF16</span>
                  </div>
                </div>
                <div className="mt-3 text-xs text-gray-400">
                  Modern GPUs have dedicated Tensor Cores for half-precision operations
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
                <h4 className="font-semibold text-green-300 mb-3">⚡ QLoRA (4-bit Training)</h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <div className="flex justify-between">
                    <span>Memory Reduction:</span>
                    <span className="text-green-400 font-semibold">4x reduction</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Trainable Params:</span>
                    <span className="text-blue-400">&lt;1% of model</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Quality Loss:</span>
                    <span className="text-green-400">&lt;1% typically</span>
                  </div>
                </div>
                <div className="mt-3 text-xs text-gray-400">
                  Quantizes base model to 4-bit, trains only LoRA adapters in FP16
                </div>
              </div>
              
              <div className="bg-orange-900/30 border border-orange-600/50 p-4 rounded-xl">
                <h4 className="font-semibold text-orange-300 mb-3">🚀 Flash Attention</h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <div className="flex justify-between">
                    <span>Memory Reduction:</span>
                    <span className="text-green-400 font-semibold">10-20x less</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Speed Improvement:</span>
                    <span className="text-green-400 font-semibold">2-4x faster</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Max Sequence:</span>
                    <span className="text-blue-400">1M+ tokens</span>
                  </div>
                </div>
                <div className="mt-3 text-xs text-gray-400">
                  Never materializes full attention matrix - computes in GPU SRAM
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-yellow-900/30 border border-yellow-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-yellow-300 mb-3">🎯 Practical Combinations</h4>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-green-400 mb-2">Consumer GPU (16GB)</h5>
              <ul className="text-gray-300 text-xs space-y-1">
                <li>• QLoRA + Gradient Checkpointing</li>
                <li>• Mixed Precision (BF16)</li>
                <li>• Flash Attention if available</li>
                <li>• <strong>Result:</strong> Train 7B models</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-blue-400 mb-2">Workstation (48GB)</h5>
              <ul className="text-gray-300 text-xs space-y-1">
                <li>• Full precision training possible</li>
                <li>• LoRA for faster iterations</li>
                <li>• Gradient checkpointing optional</li>
                <li>• <strong>Result:</strong> Train 13B models</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-purple-400 mb-2">Enterprise (80GB+)</h5>
              <ul className="text-gray-300 text-xs space-y-1">
                <li>• Full fine-tuning viable</li>
                <li>• Multiple models in parallel</li>
                <li>• Longer context windows</li>
                <li>• <strong>Result:</strong> Train 70B models</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">⚙️ Implementation Tips</h4>
            <div className="space-y-2 text-sm text-gray-300">
              <div><span className="text-blue-400">torch.compile():</span> +20% speed boost</div>
              <div><span className="text-green-400">Gradient accumulation:</span> Simulate large batches</div>
              <div><span className="text-purple-400">CPU offloading:</span> For extreme memory constraints</div>
              <div><span className="text-yellow-400">Model sharding:</span> Split across multiple GPUs</div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">🔍 Memory Monitoring</h4>
            <div className="space-y-2 text-sm text-gray-300">
              <div><span className="text-green-400">nvidia-smi:</span> Real-time GPU memory</div>
              <div><span className="text-blue-400">torch.cuda.memory_summary():</span> Detailed breakdown</div>
              <div><span className="text-yellow-400">Peak usage:</span> Monitor during validation</div>
              <div><span className="text-red-400">OOM prevention:</span> Stay &lt;90% capacity</div>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Distributed Training",
    subtitle: "Multi-GPU Scaling Strategies",
    presenterNotes: [
      "Distributed training is essential for models that don't fit on single GPU",
      "DDP (Data Parallel) is simplest - full model copy per GPU, split batches",
      "FSDP (Fully Sharded) shards model itself across GPUs for huge models",
      "Gradient accumulation simulates larger batches across time steps",
      "Pipeline parallel splits model layers across GPUs - complex but powerful",
      "Start with DDP, move to FSDP only when model doesn't fit in GPU memory"
    ],
    content: (
      <div className="space-y-6">
        <div className="space-y-4">
          <h3 className="text-xl font-bold text-white text-center">Scaling Beyond Single GPU</h3>
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
                <h4 className="font-semibold text-blue-300 mb-3">📊 Data Parallel (DDP)</h4>
                <div className="bg-gray-800/50 p-3 rounded mb-3">
                  <div className="text-xs font-mono text-gray-300">
                    GPU 0: Model Copy + Batch[0:8]<br/>
                    GPU 1: Model Copy + Batch[8:16]<br/>
                    GPU 2: Model Copy + Batch[16:24]<br/>
                    → Sync gradients → Update all
                  </div>
                </div>
                <ul className="text-sm text-gray-300 space-y-1">
                  <li>• <span className="text-green-400">✓ Simple to implement</span></li>
                  <li>• <span className="text-blue-400">✓ Linear speedup (usually)</span></li>
                  <li>• <span className="text-yellow-400">△ Full model per GPU</span></li>
                  <li>• <span className="text-red-400">✗ Memory limited by single GPU</span></li>
                </ul>
              </div>
              
              <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
                <h4 className="font-semibold text-purple-300 mb-3">🔀 Fully Sharded (FSDP)</h4>
                <div className="bg-gray-800/50 p-3 rounded mb-3">
                  <div className="text-xs font-mono text-gray-300">
                    GPU 0: Layers 0-8 + Batch[0:8]<br/>
                    GPU 1: Layers 9-16 + Batch[8:16]<br/>
                    GPU 2: Layers 17-24 + Batch[16:24]<br/>
                    → Communicate as needed
                  </div>
                </div>
                <ul className="text-sm text-gray-300 space-y-1">
                  <li>• <span className="text-green-400">✓ Enables huge models</span></li>
                  <li>• <span className="text-blue-400">✓ Memory scales with GPUs</span></li>
                  <li>• <span className="text-yellow-400">△ More communication</span></li>
                  <li>• <span className="text-red-400">✗ Complex to debug</span></li>
                </ul>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
                <h4 className="font-semibold text-green-300 mb-3">⏱️ Gradient Accumulation</h4>
                <div className="bg-gray-800/50 p-3 rounded mb-3">
                  <div className="text-xs font-mono text-gray-300">
                    effective_batch = micro_batch × <br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;accumulation_steps × <br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;num_gpus
                  </div>
                </div>
                <div className="space-y-2 text-sm text-gray-300">
                  <div className="flex justify-between">
                    <span>Memory:</span>
                    <span className="text-green-400">Same as micro_batch</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Compute:</span>
                    <span className="text-yellow-400">Longer time to convergence</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Use case:</span>
                    <span className="text-blue-400">Simulate large batches</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-orange-900/30 border border-orange-600/50 p-4 rounded-xl">
                <h4 className="font-semibold text-orange-300 mb-3">🔄 Pipeline Parallel</h4>
                <div className="bg-gray-800/50 p-3 rounded mb-3">
                  <div className="text-xs font-mono text-gray-300">
                    GPU 0: Embed + Layers 0-7<br/>
                    GPU 1: Layers 8-15<br/>
                    GPU 2: Layers 16-23 + Head<br/>
                    → Sequential flow
                  </div>
                </div>
                <ul className="text-sm text-gray-300 space-y-1">
                  <li>• <span className="text-green-400">✓ Very deep models</span></li>
                  <li>• <span className="text-blue-400">✓ Memory efficient</span></li>
                  <li>• <span className="text-yellow-400">△ Pipeline bubbles</span></li>
                  <li>• <span className="text-red-400">✗ Hard to balance</span></li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        
        <div className="bg-yellow-900/30 border border-yellow-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-yellow-300 mb-3">🎯 Strategy Decision Matrix</h4>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-blue-400 mb-2">Model Fits in Single GPU</h5>
              <ul className="text-gray-300 text-xs space-y-1">
                <li>• <strong>Use:</strong> DDP for speed</li>
                <li>• <strong>Benefit:</strong> Linear scaling</li>
                <li>• <strong>Example:</strong> 7B model on A100</li>
                <li>• <strong>Setup:</strong> <code>accelerate launch --multi_gpu</code></li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-green-400 mb-2">Model Too Large</h5>
              <ul className="text-gray-300 text-xs space-y-1">
                <li>• <strong>Use:</strong> FSDP or Model Sharding</li>
                <li>• <strong>Benefit:</strong> Enables training</li>
                <li>• <strong>Example:</strong> 70B model across 8 GPUs</li>
                <li>• <strong>Setup:</strong> DeepSpeed ZeRO Stage 3</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-purple-400 mb-2">Memory Constrained</h5>
              <ul className="text-gray-300 text-xs space-y-1">
                <li>• <strong>Use:</strong> Gradient Accumulation</li>
                <li>• <strong>Benefit:</strong> Maintains batch size</li>
                <li>• <strong>Example:</strong> Simulate batch=128 with micro=4</li>
                <li>• <strong>Setup:</strong> accumulation_steps=32</li>
              </ul>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">🛠️ Implementation Commands</h4>
            <div className="space-y-2">
              <div className="bg-gray-700 p-2 rounded">
                <div className="text-xs text-blue-400 font-semibold mb-1">Single Node, Multiple GPUs</div>
                <code className="text-green-300 text-xs">accelerate launch --multi_gpu train.py</code>
              </div>
              <div className="bg-gray-700 p-2 rounded">
                <div className="text-xs text-purple-400 font-semibold mb-1">Multiple Nodes</div>
                <code className="text-green-300 text-xs">accelerate launch --num_machines=4 train.py</code>
              </div>
              <div className="bg-gray-700 p-2 rounded">
                <div className="text-xs text-orange-400 font-semibold mb-1">DeepSpeed FSDP</div>
                <code className="text-green-300 text-xs">deepspeed --num_gpus=8 train.py --deepspeed ds_config.json</code>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">⚡ Performance Tips</h4>
            <div className="space-y-2 text-sm text-gray-300">
              <div><span className="text-blue-400">Network:</span> InfiniBand &gt; Ethernet for multi-node</div>
              <div><span className="text-green-400">Synchronization:</span> Minimize all-reduce frequency</div>
              <div><span className="text-yellow-400">Load balancing:</span> Equal work per GPU</div>
              <div><span className="text-purple-400">Monitoring:</span> Watch communication overhead</div>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Evaluation",
    subtitle: "Measuring Model Quality",
    presenterNotes: [
      "Evaluation is crucial - you can't improve what you can't measure",
      "Perplexity: Most common metric for language models, measures prediction confidence",
      "Lower perplexity = model is more confident about correct predictions",
      "BLEU/ROUGE: Standard for translation/summarization, compares to reference text",
      "Human evaluation is still gold standard - automated metrics can miss quality issues",
      "Benchmark suites like MMLU test broad knowledge, HumanEval tests coding ability",
      "Always create custom benchmarks for your specific use case",
      "Don't just rely on loss curves - generate samples and inspect them manually"
    ],
    content: (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
              <h4 className="font-semibold text-blue-300 mb-3">📊 Automatic Metrics</h4>
              <div className="space-y-3">
                <div className="bg-gray-800/50 p-3 rounded">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-white font-semibold text-sm">Perplexity</span>
                    <span className="text-green-400 text-xs">Lower is better</span>
                  </div>
                  <div className="text-xs font-mono text-gray-300 mb-2">
                    PPL = exp(avg_loss) = exp(-1/N × Σlog P(wᵢ|context))
                  </div>
                  <p className="text-gray-400 text-xs">Measures how "surprised" the model is by correct answers. PPL=10 means model considers ~10 words equally likely at each position.</p>
                </div>
                <div className="bg-gray-800/50 p-3 rounded">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-white font-semibold text-sm">BLEU / ROUGE</span>
                    <span className="text-green-400 text-xs">Higher is better</span>
                  </div>
                  <p className="text-gray-400 text-xs">N-gram overlap with reference text. BLEU for translation (precision-focused), ROUGE for summarization (recall-focused).</p>
                </div>
                <div className="bg-gray-800/50 p-3 rounded">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-white font-semibold text-sm">Task-Specific</span>
                    <span className="text-blue-400 text-xs">Varies by task</span>
                  </div>
                  <p className="text-gray-400 text-xs">Accuracy, F1, Exact Match for classification/QA. Use domain-appropriate metrics.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
              <h4 className="font-semibold text-green-300 mb-3">👥 Human Evaluation</h4>
              <div className="space-y-2 text-sm text-gray-300">
                <div className="flex items-start gap-2">
                  <span className="text-green-400">✓</span>
                  <div>
                    <span className="text-white font-semibold">Quality Assessment</span>
                    <p className="text-xs text-gray-400">Rate fluency, coherence, relevance on 1-5 scale</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400">✓</span>
                  <div>
                    <span className="text-white font-semibold">Safety Checking</span>
                    <p className="text-xs text-gray-400">Test for harmful, biased, or inappropriate outputs</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400">✓</span>
                  <div>
                    <span className="text-white font-semibold">Edge Case Testing</span>
                    <p className="text-xs text-gray-400">Adversarial prompts, unusual inputs, boundary conditions</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-green-400">✓</span>
                  <div>
                    <span className="text-white font-semibold">A/B Comparisons</span>
                    <p className="text-xs text-gray-400">Side-by-side with baseline or previous version</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
              <h4 className="font-semibold text-purple-300 mb-3">📈 Loss Curves</h4>
              <div className="text-sm text-gray-300 space-y-2">
                <div className="flex justify-between">
                  <span>Healthy training:</span>
                  <span className="text-green-400">Smooth decrease, train ≈ val</span>
                </div>
                <div className="flex justify-between">
                  <span>Overfitting:</span>
                  <span className="text-red-400">Train↓ but Val↑ or flat</span>
                </div>
                <div className="flex justify-between">
                  <span>Underfitting:</span>
                  <span className="text-yellow-400">Both high, not decreasing</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-orange-900/30 border border-orange-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-orange-300 mb-3">🏆 Benchmark Suites</h4>
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <h5 className="font-semibold text-blue-400 mb-2">MMLU</h5>
              <p className="text-xs text-gray-400 mb-2">Multitask Language Understanding</p>
              <ul className="text-xs text-gray-300 text-left space-y-1">
                <li>• 57 subjects</li>
                <li>• STEM, humanities, social</li>
                <li>• Tests broad knowledge</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <h5 className="font-semibold text-green-400 mb-2">HumanEval</h5>
              <p className="text-xs text-gray-400 mb-2">Code Generation</p>
              <ul className="text-xs text-gray-300 text-left space-y-1">
                <li>• 164 problems</li>
                <li>• Python functions</li>
                <li>• Unit test validation</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <h5 className="font-semibold text-purple-400 mb-2">BBH</h5>
              <p className="text-xs text-gray-400 mb-2">BIG-Bench Hard</p>
              <ul className="text-xs text-gray-300 text-left space-y-1">
                <li>• 23 hard tasks</li>
                <li>• Reasoning focus</li>
                <li>• Chain-of-thought</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <h5 className="font-semibold text-yellow-400 mb-2">Custom</h5>
              <p className="text-xs text-gray-400 mb-2">Domain-Specific</p>
              <ul className="text-xs text-gray-300 text-left space-y-1">
                <li>• Your use case</li>
                <li>• Real examples</li>
                <li>• Business metrics</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">🛠️ Evaluation Code Example</h4>
            <div className="bg-gray-900 p-3 rounded text-xs font-mono text-green-300 space-y-1">
              <p className="text-gray-500"># Calculate perplexity</p>
              <p>with torch.no_grad():</p>
              <p>&nbsp;&nbsp;outputs = model(input_ids, labels=input_ids)</p>
              <p>&nbsp;&nbsp;loss = outputs.loss</p>
              <p>&nbsp;&nbsp;perplexity = torch.exp(loss)</p>
              <p className="text-gray-500"># Good: PPL &lt; 20, Great: PPL &lt; 10</p>
            </div>
          </div>

          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">⚠️ Common Evaluation Mistakes</h4>
            <div className="space-y-2 text-sm text-gray-300">
              <div className="flex items-start gap-2">
                <span className="text-red-400">✗</span>
                <span>Only looking at loss, not generating samples</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-red-400">✗</span>
                <span>Test set leakage from training data</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-red-400">✗</span>
                <span>Ignoring task-specific metrics</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-red-400">✗</span>
                <span>No baseline comparison</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Validation",
    subtitle: "Avoiding Overfitting & Ensuring Generalization",
    presenterNotes: [
      "Validation is your safety net against overfitting - the model memorizing training data",
      "Data splits: 80-90% train, 5-10% validation, 5-10% test - NEVER touch test until final evaluation",
      "Data leakage is subtle but deadly - same documents, paraphrases, or related content across splits",
      "Early stopping: Stop when validation loss hasn't improved for N steps (patience)",
      "Checkpoint frequently - you can always go back to best validation checkpoint",
      "For LLMs: Also generate samples periodically to visually inspect quality",
      "Watch the train-val gap: growing gap = overfitting, both high = underfitting",
      "Quality assurance: Compare to baseline, test diverse prompts, check for regressions"
    ],
    content: (
      <div className="space-y-6">
        <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-blue-300 mb-4">📊 Data Split Strategy</h4>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-green-900/40 p-4 rounded-lg text-center">
              <div className="text-3xl font-bold text-green-300 mb-2">80-90%</div>
              <div className="text-white font-semibold mb-1">Training Set</div>
              <p className="text-xs text-gray-400">Model learns from this data. Largest portion for maximum learning signal.</p>
            </div>
            <div className="bg-yellow-900/40 p-4 rounded-lg text-center">
              <div className="text-3xl font-bold text-yellow-300 mb-2">5-10%</div>
              <div className="text-white font-semibold mb-1">Validation Set</div>
              <p className="text-xs text-gray-400">Tune hyperparameters, early stopping decisions. Check during training.</p>
            </div>
            <div className="bg-blue-900/40 p-4 rounded-lg text-center">
              <div className="text-3xl font-bold text-blue-300 mb-2">5-10%</div>
              <div className="text-white font-semibold mb-1">Test Set</div>
              <p className="text-xs text-gray-400">Final evaluation ONLY. Never use for decisions during training.</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="bg-red-900/30 border border-red-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-red-300 mb-3">🚨 Overfitting Detection</h4>
            <div className="space-y-3">
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-white font-semibold text-sm">Classic Signs</span>
                  <span className="text-red-400 text-xs">⚠️ Warning</span>
                </div>
                <ul className="text-sm text-gray-300 space-y-1">
                  <li>• Train loss ↓ but Val loss ↑ or flat</li>
                  <li>• Growing gap between train and val</li>
                  <li>• Perfect training accuracy (suspicious!)</li>
                  <li>• Model outputs training examples verbatim</li>
                </ul>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="text-white font-semibold text-sm mb-2">Visual Pattern</div>
                <div className="text-xs font-mono text-gray-300">
                  <p>Loss │</p>
                  <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│╲ val (↗ bad!)</p>
                  <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│ ╲____</p>
                  <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;╲ train (keeps ↓)</p>
                  <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──────────→ Steps</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-green-300 mb-3">✅ Prevention Strategies</h4>
            <div className="space-y-2 text-sm text-gray-300">
              <div className="flex items-start gap-2">
                <span className="text-green-400">1.</span>
                <div>
                  <span className="text-white font-semibold">Early Stopping</span>
                  <p className="text-xs text-gray-400">Stop when val loss hasn't improved for N steps (patience=3-5 evals)</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-400">2.</span>
                <div>
                  <span className="text-white font-semibold">Regular Checkpoints</span>
                  <p className="text-xs text-gray-400">Save every N steps, keep best K checkpoints by val loss</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-400">3.</span>
                <div>
                  <span className="text-white font-semibold">Regularization</span>
                  <p className="text-xs text-gray-400">Weight decay (0.01-0.1), dropout (0-0.1 for LLMs)</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-400">4.</span>
                <div>
                  <span className="text-white font-semibold">More Data</span>
                  <p className="text-xs text-gray-400">Best regularizer is more diverse training data</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-green-400">5.</span>
                <div>
                  <span className="text-white font-semibold">Reduce Model Capacity</span>
                  <p className="text-xs text-gray-400">Lower LoRA rank, fewer layers, smaller model</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-orange-900/30 border border-orange-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-orange-300 mb-3">⚠️ Data Leakage - The Silent Killer</h4>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-red-400 mb-2">Common Sources</h5>
              <ul className="text-xs text-gray-300 space-y-1">
                <li>• Same doc in train & test</li>
                <li>• Paraphrased content</li>
                <li>• Related documents (same topic)</li>
                <li>• Temporal leakage (future data)</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-yellow-400 mb-2">Detection</h5>
              <ul className="text-xs text-gray-300 space-y-1">
                <li>• N-gram overlap analysis</li>
                <li>• Embedding similarity check</li>
                <li>• Suspiciously high test scores</li>
                <li>• Model outputs test verbatim</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-green-400 mb-2">Prevention</h5>
              <ul className="text-xs text-gray-300 space-y-1">
                <li>• Split by document/source</li>
                <li>• Deduplication before split</li>
                <li>• Time-based splits if applicable</li>
                <li>• Manual spot-check samples</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">🛠️ Early Stopping Implementation</h4>
            <div className="bg-gray-900 p-3 rounded text-xs font-mono text-green-300 space-y-1">
              <p className="text-gray-500"># Trainer with early stopping</p>
              <p>training_args = TrainingArguments(</p>
              <p>&nbsp;&nbsp;evaluation_strategy="steps",</p>
              <p>&nbsp;&nbsp;eval_steps=100,</p>
              <p>&nbsp;&nbsp;save_strategy="steps",</p>
              <p>&nbsp;&nbsp;save_steps=100,</p>
              <p>&nbsp;&nbsp;load_best_model_at_end=True,</p>
              <p>&nbsp;&nbsp;metric_for_best_model="eval_loss",</p>
              <p>&nbsp;&nbsp;greater_is_better=False,</p>
              <p>)</p>
            </div>
          </div>

          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">📋 Quality Assurance Checklist</h4>
            <div className="space-y-2 text-sm text-gray-300">
              <div className="flex items-center gap-2">
                <span className="text-blue-400">□</span>
                <span>Generate diverse sample outputs</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-400">□</span>
                <span>Compare to baseline/previous version</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-400">□</span>
                <span>Test edge cases and adversarial inputs</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-400">□</span>
                <span>Check for capability regressions</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-400">□</span>
                <span>Verify no training data memorization</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Evaluation vs Validation",
    subtitle: "Understanding the Distinction",
    presenterNotes: [
      "Common confusion: these terms are often used interchangeably, but they serve different purposes",
      "Validation is a SUBSET of evaluation - it's evaluation specifically during training",
      "Validation answers: 'Is my model learning or memorizing?' - guides training decisions",
      "Evaluation answers: 'How good is my model?' - measures capabilities and quality",
      "Validation uses held-out data to make training decisions (early stopping, checkpoints)",
      "Evaluation is broader: benchmarks, human review, A/B tests, production metrics",
      "You validate DURING training, you evaluate DURING and AFTER training",
      "Think: Validation = training guardrails, Evaluation = quality measurement"
    ],
    content: (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-blue-900/30 border border-blue-600/50 p-5 rounded-xl">
            <div className="flex items-center gap-3 mb-4">
              <span className="text-3xl">📊</span>
              <div>
                <h4 className="font-bold text-blue-300 text-xl">Evaluation</h4>
                <p className="text-blue-200 text-sm">"How good is this model?"</p>
              </div>
            </div>
            <div className="space-y-3">
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">Purpose</h5>
                <p className="text-gray-300 text-sm">Measure model quality, capabilities, and fitness for task</p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">When</h5>
                <p className="text-gray-300 text-sm">During <strong>and</strong> after training</p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">Metrics</h5>
                <p className="text-gray-300 text-sm">Perplexity, BLEU, ROUGE, MMLU, HumanEval, human review</p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">Data Used</h5>
                <p className="text-gray-300 text-sm">Val set, test set, benchmarks, real-world samples</p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">Actions</h5>
                <p className="text-gray-300 text-sm">Compare models, report results, decide if production-ready</p>
              </div>
            </div>
          </div>

          <div className="bg-green-900/30 border border-green-600/50 p-5 rounded-xl">
            <div className="flex items-center gap-3 mb-4">
              <span className="text-3xl">🛡️</span>
              <div>
                <h4 className="font-bold text-green-300 text-xl">Validation</h4>
                <p className="text-green-200 text-sm">"Is my model generalizing?"</p>
              </div>
            </div>
            <div className="space-y-3">
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">Purpose</h5>
                <p className="text-gray-300 text-sm">Prevent overfitting, guide training decisions</p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">When</h5>
                <p className="text-gray-300 text-sm">During training <strong>only</strong></p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">Metrics</h5>
                <p className="text-gray-300 text-sm">Train vs Val loss gap, convergence curves, loss trends</p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">Data Used</h5>
                <p className="text-gray-300 text-sm">Validation split only (held-out from training)</p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">Actions</h5>
                <p className="text-gray-300 text-sm">Early stopping, checkpoint selection, hyperparameter tuning</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-purple-300 mb-4">🔄 The Relationship</h4>
          <div className="flex items-center justify-center gap-4">
            <div className="bg-blue-900/50 px-6 py-4 rounded-xl text-center">
              <p className="text-blue-300 font-bold text-lg">Evaluation</p>
              <p className="text-gray-400 text-xs">Umbrella term</p>
            </div>
            <div className="text-gray-400 text-2xl">⊃</div>
            <div className="bg-green-900/50 px-6 py-4 rounded-xl text-center">
              <p className="text-green-300 font-bold text-lg">Validation</p>
              <p className="text-gray-400 text-xs">Subset during training</p>
            </div>
          </div>
          <p className="text-center text-gray-300 text-sm mt-4">
            Validation is a <strong className="text-purple-300">specific type of evaluation</strong> used during training to guide the process
          </p>
        </div>

        <div className="bg-yellow-900/30 border border-yellow-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-yellow-300 mb-3">💡 Simple Mental Model</h4>
          <div className="grid grid-cols-2 gap-6">
            <div className="text-center">
              <div className="text-4xl mb-2">🚦</div>
              <p className="text-white font-semibold">Validation = Traffic Lights</p>
              <p className="text-gray-400 text-sm">Guides you during the journey (training)</p>
              <p className="text-gray-400 text-xs mt-1">"Should I stop? Keep going? Turn?"</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">🏁</div>
              <p className="text-white font-semibold">Evaluation = Race Results</p>
              <p className="text-gray-400 text-sm">Measures overall performance</p>
              <p className="text-gray-400 text-xs mt-1">"How fast? How well? Good enough?"</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 p-4 rounded-xl">
          <h4 className="font-semibold text-gray-300 mb-3">📋 Quick Reference</h4>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-600">
                <th className="text-left py-2">Question</th>
                <th className="text-center py-2">Use...</th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              <tr className="border-b border-gray-700">
                <td className="py-2">"When should I stop training?"</td>
                <td className="text-center py-2"><span className="text-green-400 font-semibold">Validation</span></td>
              </tr>
              <tr className="border-b border-gray-700">
                <td className="py-2">"Which checkpoint is best?"</td>
                <td className="text-center py-2"><span className="text-green-400 font-semibold">Validation</span></td>
              </tr>
              <tr className="border-b border-gray-700">
                <td className="py-2">"How does my model compare to GPT-4?"</td>
                <td className="text-center py-2"><span className="text-blue-400 font-semibold">Evaluation</span></td>
              </tr>
              <tr className="border-b border-gray-700">
                <td className="py-2">"Is my model ready for production?"</td>
                <td className="text-center py-2"><span className="text-blue-400 font-semibold">Evaluation</span></td>
              </tr>
              <tr>
                <td className="py-2">"Is my model overfitting?"</td>
                <td className="text-center py-2"><span className="text-green-400 font-semibold">Validation</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    )
  },
  {
    title: "Deployment",
    subtitle: "From Training to Production",
    presenterNotes: [
      "Deployment is where your trained model becomes useful - many options depending on scale",
      "vLLM: Best for high-throughput production serving, uses PagedAttention for efficiency",
      "TGI: HuggingFace's solution, great integration with their ecosystem",
      "Ollama: Perfect for local development and testing, runs models on your laptop",
      "llama.cpp: C++ implementation, runs on CPU, great for edge/embedded",
      "Open WebUI: Self-hosted ChatGPT-like interface, works with Ollama backend",
      "Hyperstac.cloud: Managed GPU cloud for inference, pay-per-use",
      "Key decision: Self-hosted vs managed, latency requirements, cost constraints"
    ],
    content: (
      <div className="space-y-6">
        <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-blue-300 mb-4">🚀 Inference Frameworks</h4>
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-gray-800/50 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">⚡</span>
                <h5 className="font-bold text-white">vLLM</h5>
              </div>
              <p className="text-xs text-gray-400 mb-2">High-throughput serving</p>
              <ul className="text-xs text-gray-300 space-y-1">
                <li>• PagedAttention</li>
                <li>• Continuous batching</li>
                <li>• OpenAI-compatible API</li>
                <li>• Best for production</li>
              </ul>
              <div className="mt-2 text-xs font-mono text-green-300 bg-gray-900 p-1 rounded">
                vllm serve model_name
              </div>
            </div>
            <div className="bg-gray-800/50 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">🤗</span>
                <h5 className="font-bold text-white">TGI</h5>
              </div>
              <p className="text-xs text-gray-400 mb-2">HuggingFace's solution</p>
              <ul className="text-xs text-gray-300 space-y-1">
                <li>• Flash Attention</li>
                <li>• Tensor parallelism</li>
                <li>• Quantization support</li>
                <li>• HF Hub integration</li>
              </ul>
              <div className="mt-2 text-xs font-mono text-green-300 bg-gray-900 p-1 rounded">
                docker run ghcr.io/hf/tgi
              </div>
            </div>
            <div className="bg-gray-800/50 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">🦙</span>
                <h5 className="font-bold text-white">Ollama</h5>
              </div>
              <p className="text-xs text-gray-400 mb-2">Local-first, easy setup</p>
              <ul className="text-xs text-gray-300 space-y-1">
                <li>• One-command install</li>
                <li>• Mac/Linux/Windows</li>
                <li>• Model library built-in</li>
                <li>• Great for dev/test</li>
              </ul>
              <div className="mt-2 text-xs font-mono text-green-300 bg-gray-900 p-1 rounded">
                ollama run llama3.2
              </div>
            </div>
            <div className="bg-gray-800/50 p-3 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">🔧</span>
                <h5 className="font-bold text-white">llama.cpp</h5>
              </div>
              <p className="text-xs text-gray-400 mb-2">Pure C++, runs anywhere</p>
              <ul className="text-xs text-gray-300 space-y-1">
                <li>• CPU inference</li>
                <li>• GGUF format</li>
                <li>• Edge/embedded</li>
                <li>• Minimal deps</li>
              </ul>
              <div className="mt-2 text-xs font-mono text-green-300 bg-gray-900 p-1 rounded">
                ./main -m model.gguf
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-green-300 mb-3">🖥️ User Interfaces</h4>
            <div className="space-y-3">
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-white font-semibold">Open WebUI</span>
                  <span className="text-green-400 text-xs">Self-hosted</span>
                </div>
                <p className="text-gray-400 text-xs mb-2">ChatGPT-like interface for local models</p>
                <ul className="text-xs text-gray-300 space-y-1">
                  <li>• Works with Ollama backend</li>
                  <li>• Multi-user support</li>
                  <li>• RAG capabilities built-in</li>
                  <li>• Docker one-liner setup</li>
                </ul>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-white font-semibold">Alternatives</span>
                </div>
                <div className="flex flex-wrap gap-1 mt-1">
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">text-generation-webui</span>
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">LM Studio</span>
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">Jan.ai</span>
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">GPT4All</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-purple-300 mb-3">☁️ Managed Platforms</h4>
            <div className="space-y-3">
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-white font-semibold">Hyperstac.cloud</span>
                  <span className="text-purple-400 text-xs">GPU Cloud</span>
                </div>
                <p className="text-gray-400 text-xs mb-2">Managed inference infrastructure</p>
                <ul className="text-xs text-gray-300 space-y-1">
                  <li>• Pay-per-use GPU</li>
                  <li>• Auto-scaling</li>
                  <li>• Deploy custom models</li>
                </ul>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-white font-semibold">Alternatives</span>
                </div>
                <div className="flex flex-wrap gap-1 mt-1">
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">Replicate</span>
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">Modal</span>
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">RunPod</span>
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">Beam</span>
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">Baseten</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-orange-900/30 border border-orange-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-orange-300 mb-3">⚙️ Performance Optimizations</h4>
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <h5 className="font-semibold text-blue-400 mb-2">Continuous Batching</h5>
              <p className="text-xs text-gray-400">Process new requests without waiting for batch to complete</p>
              <p className="text-green-400 text-xs mt-1">↑ 2-4x throughput</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <h5 className="font-semibold text-green-400 mb-2">KV Cache</h5>
              <p className="text-xs text-gray-400">Store computed attention keys/values for reuse</p>
              <p className="text-green-400 text-xs mt-1">↓ Redundant compute</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <h5 className="font-semibold text-purple-400 mb-2">Tensor Parallel</h5>
              <p className="text-xs text-gray-400">Split model across multiple GPUs</p>
              <p className="text-green-400 text-xs mt-1">↑ Larger models</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <h5 className="font-semibold text-yellow-400 mb-2">Speculative Decoding</h5>
              <p className="text-xs text-gray-400">Small model drafts, large model verifies</p>
              <p className="text-green-400 text-xs mt-1">↑ 2-3x speed</p>
            </div>
          </div>
        </div>

        <div className="bg-yellow-900/30 border border-yellow-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-yellow-300 mb-3">🎯 Deployment Decision Matrix</h4>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-600">
                <th className="text-left py-2">Scenario</th>
                <th className="text-center py-2">Recommended</th>
                <th className="text-left py-2">Why</th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              <tr className="border-b border-gray-700">
                <td className="py-2">Local dev/testing</td>
                <td className="text-center py-2"><span className="text-green-400 font-semibold">Ollama</span></td>
                <td className="py-2 text-xs">Easy setup, works on laptop</td>
              </tr>
              <tr className="border-b border-gray-700">
                <td className="py-2">Team/company chatbot</td>
                <td className="text-center py-2"><span className="text-blue-400 font-semibold">Ollama + Open WebUI</span></td>
                <td className="py-2 text-xs">Self-hosted, multi-user, private</td>
              </tr>
              <tr className="border-b border-gray-700">
                <td className="py-2">High-traffic production</td>
                <td className="text-center py-2"><span className="text-purple-400 font-semibold">vLLM / TGI</span></td>
                <td className="py-2 text-xs">Optimized throughput, batching</td>
              </tr>
              <tr className="border-b border-gray-700">
                <td className="py-2">Edge/mobile/embedded</td>
                <td className="text-center py-2"><span className="text-orange-400 font-semibold">llama.cpp</span></td>
                <td className="py-2 text-xs">Minimal deps, CPU inference</td>
              </tr>
              <tr>
                <td className="py-2">Serverless/scale-to-zero</td>
                <td className="text-center py-2"><span className="text-yellow-400 font-semibold">Replicate / Modal</span></td>
                <td className="py-2 text-xs">Pay only for usage</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    )
  },
  {
    title: "Scaling Strategies",
    subtitle: "From Demo to 1M Users",
    presenterNotes: [
      "Scaling is where engineering meets economics - every decision has cost implications",
      "Vertical scaling is simpler but has limits; horizontal is complex but unlimited",
      "Cold starts are the killer for serverless LLM deployments - models take seconds to load",
      "Caching is your best friend: prompt caching, KV caching, semantic caching, response caching",
      "Distillation: Train a smaller model to mimic your large model - 10x cheaper inference",
      "Monitor p99 latency, not average - users feel the tail latency",
      "Cost per token varies 100x between deployment options - do the math",
      "Multi-region adds complexity but may be required for latency or compliance"
    ],
    content: (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-blue-300 mb-3">📈 Vertical Scaling</h4>
            <p className="text-gray-400 text-sm mb-3">Bigger hardware, same architecture</p>
            <div className="space-y-2">
              <div className="bg-gray-800/50 p-2 rounded flex justify-between items-center">
                <span className="text-gray-300 text-sm">T4 (16GB)</span>
                <span className="text-blue-400 text-xs">~$0.50/hr</span>
              </div>
              <div className="bg-gray-800/50 p-2 rounded flex justify-between items-center">
                <span className="text-gray-300 text-sm">A10G (24GB)</span>
                <span className="text-blue-400 text-xs">~$1.00/hr</span>
              </div>
              <div className="bg-gray-800/50 p-2 rounded flex justify-between items-center">
                <span className="text-gray-300 text-sm">A100 (80GB)</span>
                <span className="text-blue-400 text-xs">~$3.00/hr</span>
              </div>
              <div className="bg-gray-800/50 p-2 rounded flex justify-between items-center">
                <span className="text-gray-300 text-sm">H100 (80GB)</span>
                <span className="text-blue-400 text-xs">~$4.50/hr</span>
              </div>
            </div>
            <div className="mt-3 text-xs text-gray-400">
              <span className="text-green-400">✓</span> Simple, no code changes<br/>
              <span className="text-red-400">✗</span> Hard limits, expensive at top
            </div>
          </div>

          <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-purple-300 mb-3">📊 Horizontal Scaling</h4>
            <p className="text-gray-400 text-sm mb-3">More instances, load balanced</p>
            <div className="space-y-2">
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="text-white font-semibold text-sm mb-1">Load Balancing</h5>
                <div className="flex flex-wrap gap-1">
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">Round-robin</span>
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">Least-connections</span>
                  <span className="bg-gray-700 px-2 py-0.5 rounded text-gray-300 text-xs">Latency-based</span>
                </div>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="text-white font-semibold text-sm mb-1">Auto-scaling Triggers</h5>
                <div className="text-xs text-gray-300 space-y-1">
                  <div className="flex justify-between">
                    <span>Queue depth</span>
                    <span className="text-purple-400">&gt; 10 requests</span>
                  </div>
                  <div className="flex justify-between">
                    <span>GPU utilization</span>
                    <span className="text-purple-400">&gt; 80%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Latency p99</span>
                    <span className="text-purple-400">&gt; 2s</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-3 text-xs text-gray-400">
              <span className="text-green-400">✓</span> Unlimited scale, redundancy<br/>
              <span className="text-red-400">✗</span> Cold starts, complexity
            </div>
          </div>
        </div>

        <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-green-300 mb-3">💾 Caching Strategies</h4>
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-blue-400 text-sm mb-2">Prompt Cache</h5>
              <p className="text-xs text-gray-400">Cache system prompts & common prefixes</p>
              <p className="text-green-400 text-xs mt-2">↓ 50% first-token latency</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-green-400 text-sm mb-2">KV Cache</h5>
              <p className="text-xs text-gray-400">Store attention keys/values across requests</p>
              <p className="text-green-400 text-xs mt-2">↓ Redundant compute</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-purple-400 text-sm mb-2">Semantic Cache</h5>
              <p className="text-xs text-gray-400">Embed queries, return similar cached responses</p>
              <p className="text-green-400 text-xs mt-2">↓ 90%+ for common queries</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-yellow-400 text-sm mb-2">Response Cache</h5>
              <p className="text-xs text-gray-400">Exact match caching for deterministic outputs</p>
              <p className="text-green-400 text-xs mt-2">↓ 100% (cache hit)</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="bg-orange-900/30 border border-orange-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-orange-300 mb-3">🔬 Model Optimization</h4>
            <div className="space-y-2">
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-white font-semibold text-sm">Distillation</span>
                  <span className="text-green-400 text-xs">10-100x cheaper</span>
                </div>
                <p className="text-xs text-gray-400">Train small model to mimic large model outputs. Trade quality for speed/cost.</p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-white font-semibold text-sm">Quantization</span>
                  <span className="text-green-400 text-xs">2-4x smaller</span>
                </div>
                <p className="text-xs text-gray-400">FP16 → INT8 → INT4. Use GPTQ, AWQ, or GGUF for serving.</p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-white font-semibold text-sm">Pruning</span>
                  <span className="text-green-400 text-xs">30-50% smaller</span>
                </div>
                <p className="text-xs text-gray-400">Remove low-importance weights. Structured pruning for real speedup.</p>
              </div>
            </div>
          </div>

          <div className="bg-yellow-900/30 border border-yellow-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-yellow-300 mb-3">📊 Metrics That Matter</h4>
            <div className="space-y-2">
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="flex justify-between items-center">
                  <span className="text-white font-semibold text-sm">Latency p99</span>
                  <span className="text-yellow-400 text-xs">Not average!</span>
                </div>
                <p className="text-xs text-gray-400">Users feel tail latency. Target: &lt;2s for chat.</p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="flex justify-between items-center">
                  <span className="text-white font-semibold text-sm">Throughput</span>
                  <span className="text-blue-400 text-xs">tokens/sec</span>
                </div>
                <p className="text-xs text-gray-400">Total tokens generated per second across all requests.</p>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <div className="flex justify-between items-center">
                  <span className="text-white font-semibold text-sm">Cost per 1K tokens</span>
                  <span className="text-green-400 text-xs">$$$</span>
                </div>
                <p className="text-xs text-gray-400">Track this! Varies 100x between options.</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 p-4 rounded-xl">
          <h4 className="font-semibold text-gray-300 mb-3">💰 Cost Comparison (7B model, 1M tokens/day)</h4>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-600">
                <th className="text-left py-2">Option</th>
                <th className="text-center py-2">Setup</th>
                <th className="text-center py-2">$/month</th>
                <th className="text-left py-2">Best for</th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              <tr className="border-b border-gray-700">
                <td className="py-2">Self-hosted (T4)</td>
                <td className="text-center py-2">Medium</td>
                <td className="text-center py-2 text-green-400">~$360</td>
                <td className="py-2 text-xs">Predictable load, technical team</td>
              </tr>
              <tr className="border-b border-gray-700">
                <td className="py-2">Serverless (Modal/Replicate)</td>
                <td className="text-center py-2">Easy</td>
                <td className="text-center py-2 text-yellow-400">~$500-800</td>
                <td className="py-2 text-xs">Variable load, fast iteration</td>
              </tr>
              <tr className="border-b border-gray-700">
                <td className="py-2">Managed (Hyperstac)</td>
                <td className="text-center py-2">Easy</td>
                <td className="text-center py-2 text-yellow-400">~$400-600</td>
                <td className="py-2 text-xs">Balance of control & convenience</td>
              </tr>
              <tr>
                <td className="py-2">API (OpenAI/Anthropic)</td>
                <td className="text-center py-2">Trivial</td>
                <td className="text-center py-2 text-red-400">~$2,000+</td>
                <td className="py-2 text-xs">Top quality, no infra</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    )
  },
  {
    title: "Training Script Structure",
    subtitle: "Workshop Boilerplate Overview",
    presenterNotes: [
      "This is the actual structure we'll use in the hands-on portion",
      "Typer CLI: Clean command-line interface with train, eval, publish commands",
      "Hydra configs: YAML-based configuration management, easy to swap models/hyperparams",
      "Makefile: One-command operations - 'make train', 'make eval', 'make publish-hf'",
      "Pipeline separates concerns: data loading, tokenization, training, evaluation",
      "PEFT/LoRA: Efficient fine-tuning, saves only adapter weights",
      "TensorBoard: Built-in logging, view with 'tensorboard --logdir outputs'",
      "LakeFS integration: Optional data versioning for production workflows"
    ],
    content: (
      <div className="space-y-6">
        <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-blue-300 mb-3">📁 Project Structure</h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-900 p-3 rounded font-mono text-xs text-gray-300">
              <p className="text-blue-400">hackathon/</p>
              <p>├── <span className="text-green-400">configs/</span></p>
              <p>│   ├── train.yaml</p>
              <p>│   ├── eval.yaml</p>
              <p>│   └── train_llama3b.yaml</p>
              <p>├── <span className="text-green-400">src/</span></p>
              <p>│   ├── main.py <span className="text-gray-500"># CLI entry</span></p>
              <p>│   ├── train/pipeline.py</p>
              <p>│   ├── eval/evaluator.py</p>
              <p>│   ├── data/loader.py</p>
              <p>│   └── deploy/push.py</p>
              <p>├── <span className="text-green-400">scripts/</span></p>
              <p>│   ├── tokenize_dataset.py</p>
              <p>│   └── merge_model.py</p>
              <p>├── <span className="text-yellow-400">Makefile</span></p>
              <p>└── requirements.txt</p>
            </div>
            <div className="space-y-3">
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-green-400 text-sm mb-2">Key Commands</h5>
                <div className="space-y-1 font-mono text-xs">
                  <p><span className="text-yellow-400">make train</span> <span className="text-gray-500"># Run training</span></p>
                  <p><span className="text-yellow-400">make eval</span> <span className="text-gray-500"># Evaluate model</span></p>
                  <p><span className="text-yellow-400">make tokenize</span> <span className="text-gray-500"># Pre-tokenize data</span></p>
                  <p><span className="text-yellow-400">make merge</span> <span className="text-gray-500"># Merge LoRA → base</span></p>
                  <p><span className="text-yellow-400">make gguf</span> <span className="text-gray-500"># Convert for Ollama</span></p>
                  <p><span className="text-yellow-400">make publish-hf</span> <span className="text-gray-500"># Push to HuggingFace</span></p>
                </div>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-purple-400 text-sm mb-2">CLI Interface</h5>
                <div className="font-mono text-xs text-green-300">
                  <p>python -m src.main train</p>
                  <p>python -m src.main eval</p>
                  <p>python -m src.main publish-hf</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-green-300 mb-3">⚙️ Hydra Config (train.yaml)</h4>
            <div className="bg-gray-900 p-3 rounded font-mono text-xs text-green-300 space-y-0.5">
              <p><span className="text-blue-400">model_name:</span> meta-llama/Llama-3.1-8B</p>
              <p><span className="text-blue-400">dataset_path:</span> ./data</p>
              <p><span className="text-blue-400">epochs:</span> 5</p>
              <p><span className="text-blue-400">batch_size:</span> 4</p>
              <p><span className="text-blue-400">lr:</span> 2e-4</p>
              <p><span className="text-blue-400">gradient_accumulation_steps:</span> 2</p>
              <p><span className="text-gray-500"># LoRA settings</span></p>
              <p><span className="text-blue-400">lora_r:</span> 64</p>
              <p><span className="text-blue-400">lora_alpha:</span> 128</p>
              <p><span className="text-blue-400">lora_target_modules:</span></p>
              <p>  - q_proj, k_proj, v_proj, o_proj</p>
            </div>
          </div>

          <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-purple-300 mb-3">🔄 Training Pipeline Flow</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="bg-blue-600 text-white px-2 py-0.5 rounded text-xs">1</span>
                <span className="text-gray-300">Load config via Hydra</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="bg-blue-600 text-white px-2 py-0.5 rounded text-xs">2</span>
                <span className="text-gray-300">Load & tokenize dataset</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="bg-blue-600 text-white px-2 py-0.5 rounded text-xs">3</span>
                <span className="text-gray-300">Load base model + apply LoRA</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="bg-blue-600 text-white px-2 py-0.5 rounded text-xs">4</span>
                <span className="text-gray-300">Train with HF Trainer</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="bg-blue-600 text-white px-2 py-0.5 rounded text-xs">5</span>
                <span className="text-gray-300">Save adapter + metrics</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="bg-blue-600 text-white px-2 py-0.5 rounded text-xs">6</span>
                <span className="text-gray-300">Update 'latest' symlink</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-orange-900/30 border border-orange-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-orange-300 mb-3">🛠️ MLOps Stack</h4>
          <div className="grid grid-cols-5 gap-3">
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <div className="text-2xl mb-1">🎯</div>
              <h5 className="font-semibold text-blue-400 text-sm">Typer</h5>
              <p className="text-xs text-gray-400">CLI framework</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <div className="text-2xl mb-1">⚙️</div>
              <h5 className="font-semibold text-green-400 text-sm">Hydra</h5>
              <p className="text-xs text-gray-400">Config management</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <div className="text-2xl mb-1">🤗</div>
              <h5 className="font-semibold text-yellow-400 text-sm">PEFT</h5>
              <p className="text-xs text-gray-400">LoRA adapters</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <div className="text-2xl mb-1">📊</div>
              <h5 className="font-semibold text-purple-400 text-sm">TensorBoard</h5>
              <p className="text-xs text-gray-400">Metrics logging</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <div className="text-2xl mb-1">🗄️</div>
              <h5 className="font-semibold text-red-400 text-sm">LakeFS</h5>
              <p className="text-xs text-gray-400">Data versioning</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">📦 Key Dependencies</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="text-gray-300"><span className="text-blue-400">torch</span> ≥2.1.0</div>
              <div className="text-gray-300"><span className="text-blue-400">transformers</span> ≥4.44.0</div>
              <div className="text-gray-300"><span className="text-blue-400">peft</span> ≥0.11.0</div>
              <div className="text-gray-300"><span className="text-blue-400">datasets</span> ≥3.0.0</div>
              <div className="text-gray-300"><span className="text-blue-400">accelerate</span> ≥0.31.0</div>
              <div className="text-gray-300"><span className="text-blue-400">hydra-core</span> ==1.3.2</div>
              <div className="text-gray-300"><span className="text-blue-400">evaluate</span> ≥0.4.0</div>
              <div className="text-gray-300"><span className="text-blue-400">tensorboard</span> ≥2.16.2</div>
            </div>
          </div>

          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">🚀 Quick Start</h4>
            <div className="bg-gray-900 p-3 rounded font-mono text-xs text-green-300 space-y-1">
              <p><span className="text-gray-500"># Clone & setup</span></p>
              <p>git clone &lt;repo&gt; && cd hackathon</p>
              <p>make venv && source .venv/bin/activate</p>
              <p>make install</p>
              <p><span className="text-gray-500"># Train!</span></p>
              <p>make train TRAIN_CONFIG=train</p>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Experiment Tracking",
    subtitle: "Logging, Comparing & Reproducing Runs",
    presenterNotes: [
      "Experiment tracking is essential - without it you'll forget what worked and why",
      "Our boilerplate uses TensorBoard by default, W&B optional for cloud sync",
      "Every run gets a timestamped folder + 'latest' symlink for easy access",
      "Config snapshots: We save the exact config used in each run for reproducibility",
      "Metrics saved as JSON for easy parsing and comparison",
      "Evaluation logs predictions.jsonl so you can inspect actual model outputs",
      "Compare runs with 'tensorboard --logdir outputs/runs' to see all experiments",
      "Pro tip: Also log git commit hash to track code version"
    ],
    content: (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-blue-300 mb-3">📊 What to Log</h4>
            <div className="space-y-3">
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">Training Metrics</h5>
                <div className="grid grid-cols-2 gap-1 text-xs text-gray-300">
                  <span>• train/loss</span>
                  <span>• eval/loss</span>
                  <span>• learning_rate</span>
                  <span>• gradient_norm</span>
                  <span>• epoch</span>
                  <span>• global_step</span>
                </div>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">Evaluation Metrics</h5>
                <div className="grid grid-cols-2 gap-1 text-xs text-gray-300">
                  <span>• perplexity</span>
                  <span>• exact_match</span>
                  <span>• ROUGE-L</span>
                  <span>• BLEU</span>
                  <span>• jaccard_overlap</span>
                  <span>• samples_evaluated</span>
                </div>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">System Metrics</h5>
                <div className="grid grid-cols-2 gap-1 text-xs text-gray-300">
                  <span>• GPU memory used</span>
                  <span>• GPU utilization</span>
                  <span>• throughput (samples/s)</span>
                  <span>• training time</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-green-300 mb-3">📁 Run Structure</h4>
            <div className="bg-gray-900 p-3 rounded font-mono text-xs text-gray-300 mb-3">
              <p className="text-blue-400">outputs/runs/</p>
              <p>├── 2024-12-19_14-30-00/</p>
              <p>│   ├── <span className="text-green-400">model/</span> <span className="text-gray-500"># LoRA adapter</span></p>
              <p>│   ├── <span className="text-yellow-400">config_used.yaml</span></p>
              <p>│   ├── <span className="text-yellow-400">metrics.json</span></p>
              <p>│   ├── <span className="text-purple-400">logs/</span> <span className="text-gray-500"># TensorBoard</span></p>
              <p>│   └── checkpoint-*/</p>
              <p>├── 2024-12-19_16-45-00/</p>
              <p>│   └── ...</p>
              <p>└── <span className="text-cyan-400">latest</span> → 2024-12-19_16-45-00/</p>
            </div>
            <div className="bg-gray-800/50 p-2 rounded">
              <p className="text-xs text-gray-400"><span className="text-cyan-400">latest</span> symlink always points to most recent run - use for scripts!</p>
            </div>
          </div>
        </div>

        <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-purple-300 mb-3">🛠️ Tracking Tools Comparison</h4>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gray-800/50 p-3 rounded">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">📊</span>
                <h5 className="font-bold text-orange-400">TensorBoard</h5>
              </div>
              <p className="text-xs text-gray-400 mb-2">Built into our boilerplate</p>
              <ul className="text-xs text-gray-300 space-y-1">
                <li><span className="text-green-400">✓</span> Free, local, no signup</li>
                <li><span className="text-green-400">✓</span> Native HF Trainer support</li>
                <li><span className="text-green-400">✓</span> Compare multiple runs</li>
                <li><span className="text-yellow-400">△</span> No cloud sync</li>
              </ul>
              <div className="mt-2 text-xs font-mono text-green-300 bg-gray-900 p-1 rounded">
                tensorboard --logdir outputs
              </div>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">🪄</span>
                <h5 className="font-bold text-yellow-400">Weights & Biases</h5>
              </div>
              <p className="text-xs text-gray-400 mb-2">Optional in requirements.txt</p>
              <ul className="text-xs text-gray-300 space-y-1">
                <li><span className="text-green-400">✓</span> Cloud dashboard</li>
                <li><span className="text-green-400">✓</span> Team collaboration</li>
                <li><span className="text-green-400">✓</span> Hyperparameter sweeps</li>
                <li><span className="text-yellow-400">△</span> Requires account</li>
              </ul>
              <div className="mt-2 text-xs font-mono text-green-300 bg-gray-900 p-1 rounded">
                wandb.init(project="llm")
              </div>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xl">🔬</span>
                <h5 className="font-bold text-blue-400">MLflow</h5>
              </div>
              <p className="text-xs text-gray-400 mb-2">Alternative option</p>
              <ul className="text-xs text-gray-300 space-y-1">
                <li><span className="text-green-400">✓</span> Model registry</li>
                <li><span className="text-green-400">✓</span> Self-hosted option</li>
                <li><span className="text-green-400">✓</span> Deployment integration</li>
                <li><span className="text-yellow-400">△</span> More setup required</li>
              </ul>
              <div className="mt-2 text-xs font-mono text-green-300 bg-gray-900 p-1 rounded">
                mlflow.autolog()
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="bg-orange-900/30 border border-orange-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-orange-300 mb-3">🔄 Reproducibility Checklist</h4>
            <div className="space-y-2 text-sm text-gray-300">
              <div className="flex items-center gap-2">
                <span className="text-green-400">✓</span>
                <span>Config snapshot saved with each run</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-400">✓</span>
                <span>Random seeds fixed (torch, numpy)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-400">□</span>
                <span>Git commit hash logged</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-400">□</span>
                <span>requirements.txt / lock file</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-400">□</span>
                <span>Data version (LakeFS branch/commit)</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 p-4 rounded-xl">
            <h4 className="font-semibold text-gray-300 mb-3">📋 Saved Artifacts</h4>
            <div className="space-y-2">
              <div className="bg-gray-900 p-2 rounded flex justify-between items-center">
                <span className="text-sm text-gray-300">config_used.yaml</span>
                <span className="text-xs text-green-400">Hydra config snapshot</span>
              </div>
              <div className="bg-gray-900 p-2 rounded flex justify-between items-center">
                <span className="text-sm text-gray-300">metrics.json</span>
                <span className="text-xs text-blue-400">Final eval metrics</span>
              </div>
              <div className="bg-gray-900 p-2 rounded flex justify-between items-center">
                <span className="text-sm text-gray-300">predictions.jsonl</span>
                <span className="text-xs text-purple-400">Model outputs for inspection</span>
              </div>
              <div className="bg-gray-900 p-2 rounded flex justify-between items-center">
                <span className="text-sm text-gray-300">model/adapter_*.safetensors</span>
                <span className="text-xs text-yellow-400">LoRA weights</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-yellow-900/30 border border-yellow-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-yellow-300 mb-3">💡 Pro Tips</h4>
          <div className="grid grid-cols-3 gap-4 text-sm text-gray-300">
            <div>
              <span className="text-yellow-400 font-semibold">Compare runs:</span>
              <p className="text-xs mt-1">TensorBoard shows all runs in outputs/runs - easy A/B comparison</p>
            </div>
            <div>
              <span className="text-yellow-400 font-semibold">Tag experiments:</span>
              <p className="text-xs mt-1">Use config names like train_lora64.yaml, train_lora128.yaml</p>
            </div>
            <div>
              <span className="text-yellow-400 font-semibold">Inspect outputs:</span>
              <p className="text-xs mt-1">predictions.jsonl shows actual model generations vs expected</p>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Profiling",
    subtitle: "Finding Bottlenecks",
    presenterNotes: [
      "Profiling is for when training is slower than expected - find out why",
      "nvidia-smi is your first stop - run it in watch mode during training",
      "Low GPU utilization usually means data loading is the bottleneck",
      "High memory but low compute = batch size too small or I/O bound",
      "PyTorch Profiler integrates with TensorBoard for flame graphs",
      "Most common fix: increase dataloader workers or prefetch factor",
      "Don't optimize prematurely - profile only if you have a problem"
    ],
    content: (
      <div className="space-y-6">
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-blue-300 mb-3">🔍 Quick Diagnostics</h4>
            <div className="space-y-3">
              <div className="bg-gray-900 p-3 rounded">
                <p className="text-xs text-gray-400 mb-1">Watch GPU in real-time:</p>
                <code className="text-green-300 text-sm">watch -n 1 nvidia-smi</code>
              </div>
              <div className="bg-gray-800/50 p-3 rounded">
                <h5 className="font-semibold text-white text-sm mb-2">What to Look For</h5>
                <div className="space-y-1 text-xs text-gray-300">
                  <div className="flex justify-between">
                    <span>GPU Utilization</span>
                    <span className="text-green-400">Target: 90%+</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Memory Usage</span>
                    <span className="text-blue-400">Near max = good</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Temperature</span>
                    <span className="text-yellow-400">&lt;85°C safe</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-900/30 border border-green-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-green-300 mb-3">🛠️ Tools</h4>
            <div className="space-y-2">
              <div className="bg-gray-800/50 p-2 rounded flex justify-between items-center">
                <span className="text-white text-sm">nvidia-smi</span>
                <span className="text-xs text-gray-400">GPU stats, memory, temp</span>
              </div>
              <div className="bg-gray-800/50 p-2 rounded flex justify-between items-center">
                <span className="text-white text-sm">nvtop / gpustat</span>
                <span className="text-xs text-gray-400">Better TUI for GPUs</span>
              </div>
              <div className="bg-gray-800/50 p-2 rounded flex justify-between items-center">
                <span className="text-white text-sm">PyTorch Profiler</span>
                <span className="text-xs text-gray-400">Detailed op-level timing</span>
              </div>
              <div className="bg-gray-800/50 p-2 rounded flex justify-between items-center">
                <span className="text-white text-sm">htop</span>
                <span className="text-xs text-gray-400">CPU & memory overview</span>
              </div>
              <div className="bg-gray-800/50 p-2 rounded flex justify-between items-center">
                <span className="text-white text-sm">torch.profiler</span>
                <span className="text-xs text-gray-400">Memory breakdown, kernel times</span>
              </div>
              <div className="bg-gray-800/50 p-2 rounded flex justify-between items-center">
                <span className="text-white text-sm">wandb.log()</span>
                <span className="text-xs text-gray-400">Gradient norms, memory usage</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-orange-900/30 border border-orange-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-orange-300 mb-3">⚠️ Common Bottlenecks & Fixes</h4>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-red-400 text-sm mb-2">Low GPU Utilization</h5>
              <p className="text-xs text-gray-400 mb-2">GPU waiting for data</p>
              <p className="text-xs text-green-300">Fix: ↑ dataloader_num_workers</p>
              <p className="text-xs text-green-300">Fix: ↑ prefetch_factor</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-yellow-400 text-sm mb-2">OOM Errors</h5>
              <p className="text-xs text-gray-400 mb-2">Not enough VRAM</p>
              <p className="text-xs text-green-300">Fix: ↓ batch_size</p>
              <p className="text-xs text-green-300">Fix: gradient_checkpointing</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <h5 className="font-semibold text-blue-400 text-sm mb-2">Slow Throughput</h5>
              <p className="text-xs text-gray-400 mb-2">Low samples/second</p>
              <p className="text-xs text-green-300">Fix: ↑ batch_size (if VRAM ok)</p>
              <p className="text-xs text-green-300">Fix: Pre-tokenize dataset</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 p-4 rounded-xl">
          <h4 className="font-semibold text-gray-300 mb-3">💡 Our Boilerplate Optimizations</h4>
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div className="text-center">
              <code className="text-blue-400 text-xs">dataloader_num_workers: 8</code>
              <p className="text-xs text-gray-400 mt-1">Parallel data loading</p>
            </div>
            <div className="text-center">
              <code className="text-green-400 text-xs">prefetch_factor: 4</code>
              <p className="text-xs text-gray-400 mt-1">Pre-fetch batches</p>
            </div>
            <div className="text-center">
              <code className="text-purple-400 text-xs">pin_memory: true</code>
              <p className="text-xs text-gray-400 mt-1">Faster CPU→GPU transfer</p>
            </div>
            <div className="text-center">
              <code className="text-yellow-400 text-xs">make tokenize</code>
              <p className="text-xs text-gray-400 mt-1">Pre-tokenize once</p>
            </div>
          </div>
        </div>
      </div>
    )
  },
  {
    title: "Key Takeaways",
    subtitle: "What We Learned & What's Next",
    presenterNotes: [
      "Recap the journey: from transformer theory to practical training pipeline",
      "Emphasize: You don't need massive resources - LoRA + consumer GPU works",
      "The biggest insight: Fine-tuning is accessible, start small and iterate",
      "Hands-on lab is where theory becomes practice - encourage experimentation",
      "Learning path: Start with inference (Ollama), then fine-tuning, then pre-training",
      "Community resources: HuggingFace, r/LocalLLaMA, papers with code",
      "Encourage: Train on your own data, solve your own problems",
      "Final thought: AI engineering is software engineering + data + experimentation"
    ],
    content: (
      <div className="space-y-6">
        <div className="bg-blue-900/30 border border-blue-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-blue-300 mb-4">🎯 The Big Lessons</h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <span className="text-2xl">🏗️</span>
                <div>
                  <p className="text-white font-semibold">Transformers Won</p>
                  <p className="text-xs text-gray-400">Attention + parallelization beat RNNs. Decoder-only architecture is the foundation of modern LLMs.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-2xl">💡</span>
                <div>
                  <p className="text-white font-semibold">LoRA Makes It Accessible</p>
                  <p className="text-xs text-gray-400">Train only 0.1-1% of parameters. A 24GB GPU can fine-tune 7B+ models. No data center required.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-2xl">📊</span>
                <div>
                  <p className="text-white font-semibold">Data Quality &gt; Quantity</p>
                  <p className="text-xs text-gray-400">100 high-quality examples often beat 10,000 noisy ones. Curate your training data carefully.</p>
                </div>
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <span className="text-2xl">⚙️</span>
                <div>
                  <p className="text-white font-semibold">MLOps Is Essential</p>
                  <p className="text-xs text-gray-400">Config management, experiment tracking, reproducibility. Treat ML like production software.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-2xl">🔄</span>
                <div>
                  <p className="text-white font-semibold">Iterate Fast</p>
                  <p className="text-xs text-gray-400">Start small (0.5B-1B models), validate quickly, scale up only when it works.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-2xl">🎯</span>
                <div>
                  <p className="text-white font-semibold">Evaluation Matters</p>
                  <p className="text-xs text-gray-400">Don't just trust loss curves. Generate samples, test edge cases, compare to baseline.</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-green-900/50 to-emerald-900/50 border border-green-600/50 p-4 rounded-xl">
          <h4 className="font-semibold text-green-300 mb-3">🚀 Hands-On Lab: What We'll Do Next</h4>
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <div className="text-2xl mb-2">1️⃣</div>
              <p className="text-white font-semibold text-sm">Clone & Setup</p>
              <p className="text-xs text-gray-400 mt-1">Get the boilerplate running, install dependencies</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <div className="text-2xl mb-2">2️⃣</div>
              <p className="text-white font-semibold text-sm">Prepare Data</p>
              <p className="text-xs text-gray-400 mt-1">Format your dataset, run tokenization</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <div className="text-2xl mb-2">3️⃣</div>
              <p className="text-white font-semibold text-sm">Train Model</p>
              <p className="text-xs text-gray-400 mt-1">Run make train, monitor in TensorBoard</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded text-center">
              <div className="text-2xl mb-2">4️⃣</div>
              <p className="text-white font-semibold text-sm">Evaluate & Deploy</p>
              <p className="text-xs text-gray-400 mt-1">Test outputs, convert to GGUF, run in Ollama</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <div className="bg-purple-900/30 border border-purple-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-purple-300 mb-3">📚 Your Learning Path</h4>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="bg-green-600 text-white px-2 py-0.5 rounded text-xs">Now</span>
                <span className="text-gray-300 text-sm">Fine-tune with LoRA (this workshop)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="bg-blue-600 text-white px-2 py-0.5 rounded text-xs">Next</span>
                <span className="text-gray-300 text-sm">Deploy & serve (Ollama, vLLM)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="bg-purple-600 text-white px-2 py-0.5 rounded text-xs">Then</span>
                <span className="text-gray-300 text-sm">RAG & agents (retrieval augmentation)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="bg-orange-600 text-white px-2 py-0.5 rounded text-xs">Later</span>
                <span className="text-gray-300 text-sm">RLHF, DPO, preference tuning</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="bg-red-600 text-white px-2 py-0.5 rounded text-xs">Advanced</span>
                <span className="text-gray-300 text-sm">Pre-training, multi-node, custom architectures</span>
              </div>
            </div>
          </div>

          <div className="bg-orange-900/30 border border-orange-600/50 p-4 rounded-xl">
            <h4 className="font-semibold text-orange-300 mb-3">🔗 Resources to Explore</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-blue-400">🤗</span>
                <span className="text-gray-300">HuggingFace Hub - Models, datasets, spaces</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-orange-400">📖</span>
                <span className="text-gray-300">r/LocalLLaMA - Community, benchmarks, tips</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-400">📄</span>
                <span className="text-gray-300">Papers With Code - Latest research</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-purple-400">🎓</span>
                <span className="text-gray-300">Andrej Karpathy's videos - Deep understanding</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-yellow-400">💬</span>
                <span className="text-gray-300">Discord: Nous Research, EleutherAI</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-900/50 via-purple-900/50 to-pink-900/50 p-4 rounded-xl text-center">
          <p className="text-xl font-bold text-white mb-2">🧠 AI Engineering = Software Engineering + Data + Experimentation</p>
          <p className="text-gray-300">You already have the engineering skills. Now add the ML intuition through practice.</p>
        </div>
      </div>
    )
  },
  {
    title: "Essential Learning Resources",
    subtitle: "Books, Courses, Communities & People to Follow",
    presenterNotes: [
      "This slide is your roadmap to continue learning after this workshop",
      "Books: Start with Goodfellow for theory, Chollet for practical",
      "Courses: Karpathy for deep understanding, Fast.ai for practical skills",
      "Communities: Reddit r/LocalLLaMA is where the real experimentation happens",
      "Twitter/X: These people share cutting-edge insights before they hit papers",
      "Blogs: Sebastian Raschka has the best technical deep-dives",
      "Discord communities are where you get real-time help when stuck"
    ],
    content: (
      <div className="space-y-6">
        {/* Books Section */}
        <div className="bg-blue-900/20 border border-blue-600/30 rounded-xl p-5">
          <h3 className="text-xl font-bold text-blue-300 mb-4">📚 Fundamental Books</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-800/30 rounded-lg p-4">
              <h4 className="text-blue-200 font-semibold mb-2">Theory & Foundations</h4>
              <ul className="text-gray-300 text-sm space-y-2">
                <li><strong className="text-white">"Deep Learning"</strong> by Goodfellow, Bengio, Courville</li>
                <li><strong className="text-white">"Pattern Recognition and ML"</strong> by Bishop</li>
                <li><strong className="text-white">"Information Theory"</strong> by MacKay</li>
                <li><strong className="text-white">"Attention Is All You Need"</strong> paper (must read)</li>
              </ul>
            </div>
            <div className="bg-blue-800/30 rounded-lg p-4">
              <h4 className="text-blue-200 font-semibold mb-2">Practical Implementation</h4>
              <ul className="text-gray-300 text-sm space-y-2">
                <li><strong className="text-white">"Deep Learning with Python"</strong> by Chollet</li>
                <li><strong className="text-white">"Hands-On ML"</strong> by Géron</li>
                <li><strong className="text-white">"Building LLMs from Scratch"</strong> by Raschka</li>
                <li><strong className="text-white">"Natural Language Processing with Transformers"</strong></li>
              </ul>
            </div>
          </div>
        </div>

        {/* Courses Section */}
        <div className="bg-green-900/20 border border-green-600/30 rounded-xl p-5">
          <h3 className="text-xl font-bold text-green-300 mb-4">🎓 Essential Courses</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-green-800/30 rounded-lg p-4">
              <h4 className="text-green-200 font-semibold mb-2">Deep Understanding</h4>
              <ul className="text-gray-300 text-sm space-y-2">
                <li>🔥 <strong className="text-white">Andrej Karpathy's "Neural Networks: Zero to Hero"</strong></li>
                <li>🧠 <strong className="text-white">CS231n Stanford</strong> - Computer Vision</li>
                <li>📖 <strong className="text-white">CS224n Stanford</strong> - NLP with Deep Learning</li>
                <li>⚡ <strong className="text-white">"Let's build GPT"</strong> by Karpathy</li>
              </ul>
            </div>
            <div className="bg-green-800/30 rounded-lg p-4">
              <h4 className="text-green-200 font-semibold mb-2">Practical Skills</h4>
              <ul className="text-gray-300 text-sm space-y-2">
                <li>🚀 <strong className="text-white">Fast.ai</strong> - Practical Deep Learning</li>
                <li>🤗 <strong className="text-white">HuggingFace Course</strong> - Transformers</li>
                <li>🔧 <strong className="text-white">DeepLearning.ai</strong> - MLOps Specialization</li>
                <li>💻 <strong className="text-white">Full Stack Deep Learning</strong></li>
              </ul>
            </div>
          </div>
        </div>

        {/* Communities & Forums */}
        <div className="bg-purple-900/20 border border-purple-600/30 rounded-xl p-5">
          <h3 className="text-xl font-bold text-purple-300 mb-4">💬 Active Communities</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-purple-800/30 rounded-lg p-4">
              <h4 className="text-purple-200 font-semibold mb-2">Reddit</h4>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• <strong className="text-orange-400">r/LocalLLaMA</strong> - Local model running</li>
                <li>• <strong className="text-blue-400">r/MachineLearning</strong> - Research discussions</li>
                <li>• <strong className="text-green-400">r/OpenAI</strong> - API & GPT discussions</li>
                <li>• <strong className="text-yellow-400">r/MediaSynthesis</strong> - Creative AI</li>
              </ul>
            </div>
            <div className="bg-purple-800/30 rounded-lg p-4">
              <h4 className="text-purple-200 font-semibold mb-2">Discord Servers</h4>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• <strong className="text-red-400">Nous Research</strong> - Model training</li>
                <li>• <strong className="text-blue-400">EleutherAI</strong> - Open research</li>
                <li>• <strong className="text-green-400">HuggingFace</strong> - Community support</li>
                <li>• <strong className="text-orange-400">AI/ML</strong> - General discussions</li>
              </ul>
            </div>
            <div className="bg-purple-800/30 rounded-lg p-4">
              <h4 className="text-purple-200 font-semibold mb-2">Professional</h4>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• <strong className="text-blue-400">Papers With Code</strong> - Research tracker</li>
                <li>• <strong className="text-green-400">ArXiv Sanity</strong> - Paper discovery</li>
                <li>• <strong className="text-purple-400">MLOps Community</strong> - Production ML</li>
                <li>• <strong className="text-yellow-400">AI Discord</strong> - Real-time help</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Blogs & Newsletters */}
        <div className="bg-orange-900/20 border border-orange-600/30 rounded-xl p-5">
          <h3 className="text-xl font-bold text-orange-300 mb-4">📝 Must-Read Blogs</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-orange-800/30 rounded-lg p-4">
              <h4 className="text-orange-200 font-semibold mb-2">Technical Deep Dives</h4>
              <ul className="text-gray-300 text-sm space-y-2">
                <li>🔬 <strong className="text-white">Sebastian Raschka</strong> - sebastianraschka.com</li>
                <li>🧠 <strong className="text-white">Lilian Weng</strong> - lilianweng.github.io</li>
                <li>⚡ <strong className="text-white">Jay Alammar</strong> - jalammar.github.io</li>
                <li>🏗️ <strong className="text-white">Chip Huyen</strong> - huyenchip.com</li>
              </ul>
            </div>
            <div className="bg-orange-800/30 rounded-lg p-4">
              <h4 className="text-orange-200 font-semibold mb-2">Industry Insights</h4>
              <ul className="text-gray-300 text-sm space-y-2">
                <li>📊 <strong className="text-white">The Batch</strong> - DeepLearning.ai newsletter</li>
                <li>📈 <strong className="text-white">Import AI</strong> - Jack Clark's newsletter</li>
                <li>🔍 <strong className="text-white">AI Research</strong> - Paper summaries</li>
                <li>💡 <strong className="text-white">Towards Data Science</strong> - Medium</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Influencers & Researchers */}
        <div className="bg-red-900/20 border border-red-600/30 rounded-xl p-5">
          <h3 className="text-xl font-bold text-red-300 mb-4">👥 Key People to Follow</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-red-800/30 rounded-lg p-4">
              <h4 className="text-red-200 font-semibold mb-2">Researchers</h4>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• <strong className="text-white">Andrej Karpathy</strong> - @karpathy</li>
                <li>• <strong className="text-white">Yann LeCun</strong> - @ylecun</li>
                <li>• <strong className="text-white">Geoffrey Hinton</strong> - @geoffreyhinton</li>
                <li>• <strong className="text-white">Yoshua Bengio</strong> - @yoshuabengio</li>
              </ul>
            </div>
            <div className="bg-red-800/30 rounded-lg p-4">
              <h4 className="text-red-200 font-semibold mb-2">Practitioners</h4>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• <strong className="text-white">Jeremy Howard</strong> - @jeremyphoward</li>
                <li>• <strong className="text-white">Rachel Thomas</strong> - @math_rachel</li>
                <li>• <strong className="text-white">Sebastian Raschka</strong> - @rasbt</li>
                <li>• <strong className="text-white">Hugging Face</strong> - @huggingface</li>
              </ul>
            </div>
            <div className="bg-red-800/30 rounded-lg p-4">
              <h4 className="text-red-200 font-semibold mb-2">Industry Leaders</h4>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• <strong className="text-white">Chip Huyen</strong> - @chiphuyen</li>
                <li>• <strong className="text-white">Shreya Shankar</strong> - @sh_reya</li>
                <li>• <strong className="text-white">Eugene Yan</strong> - @eugeneyan</li>
                <li>• <strong className="text-white">Santiago</strong> - @svpino</li>
              </ul>
            </div>
          </div>
        </div>

        {/* YouTube Channels */}
        <div className="bg-gray-800/50 rounded-xl p-5 border border-gray-600/30">
          <h3 className="text-xl font-bold text-gray-200 mb-4">📺 YouTube Channels</h3>
          <div className="grid grid-cols-4 gap-3">
            <div className="text-center">
              <div className="text-red-400 font-semibold">🧠 Andrej Karpathy</div>
              <div className="text-xs text-gray-400">Neural networks from scratch</div>
            </div>
            <div className="text-center">
              <div className="text-blue-400 font-semibold">📊 3Blue1Brown</div>
              <div className="text-xs text-gray-400">Visual math explanations</div>
            </div>
            <div className="text-center">
              <div className="text-green-400 font-semibold">🔬 Two Minute Papers</div>
              <div className="text-xs text-gray-400">Latest AI research</div>
            </div>
            <div className="text-center">
              <div className="text-purple-400 font-semibold">⚡ Yannic Kilcher</div>
              <div className="text-xs text-gray-400">Paper reviews & analysis</div>
            </div>
          </div>
        </div>

        {/* Action Items */}
        <div className="bg-gradient-to-r from-amber-900/50 to-orange-900/50 rounded-xl p-5 border border-amber-500/30">
          <h3 className="text-xl font-bold text-amber-300 mb-3">🎯 Next Steps</h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <h4 className="text-amber-200 font-semibold mb-2">This Week:</h4>
              <ul className="text-gray-300 space-y-1">
                <li>• Join r/LocalLLaMA</li>
                <li>• Start Karpathy's videos</li>
                <li>• Set up local environment</li>
              </ul>
            </div>
            <div>
              <h4 className="text-orange-200 font-semibold mb-2">This Month:</h4>
              <ul className="text-gray-300 space-y-1">
                <li>• Complete Fast.ai course</li>
                <li>• Train your first LoRA</li>
                <li>• Join Discord communities</li>
              </ul>
            </div>
            <div>
              <h4 className="text-red-200 font-semibold mb-2">This Quarter:</h4>
              <ul className="text-gray-300 space-y-1">
                <li>• Build production system</li>
                <li>• Contribute to open source</li>
                <li>• Share your learnings</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Final Message */}
        <div className="bg-gradient-to-r from-blue-900/50 via-purple-900/50 to-pink-900/50 p-4 rounded-xl text-center">
          <p className="text-xl font-bold text-white mb-2">🚀 The Journey Starts Now</p>
          <p className="text-gray-300">You have the tools, knowledge, and resources. Time to build something amazing.</p>
          <p className="text-blue-300 text-sm mt-2 font-semibold">Stay curious. Keep building. Share what you learn.</p>
        </div>
      </div>
    )
  }
];

// localStorage key for presenter notes
const NOTES_STORAGE_KEY = 'llm-training-presenter-notes';

// Helper to get notes for a slide (localStorage > hardcoded fallback)
const getNotesForSlide = (slideIndex: number, savedNotes: Record<number, string[]>): string[] => {
  // Check localStorage first
  if (savedNotes[slideIndex]) {
    return savedNotes[slideIndex];
  }
  // Fall back to hardcoded notes
  return slides[slideIndex]?.presenterNotes || [
    'Discuss the key points on this slide',
    'Engage with audience questions',
    'Transition to next topic'
  ];
};

// Load all saved notes from localStorage
const loadSavedNotes = (): Record<number, string[]> => {
  try {
    const saved = localStorage.getItem(NOTES_STORAGE_KEY);
    return saved ? JSON.parse(saved) : {};
  } catch {
    return {};
  }
};

// Save notes to localStorage
const saveNotesToStorage = (notes: Record<number, string[]>) => {
  try {
    localStorage.setItem(NOTES_STORAGE_KEY, JSON.stringify(notes));
  } catch (e) {
    console.error('Failed to save presenter notes:', e);
  }
};

// Editable Note Component
function EditableNote({
  note,
  index,
  onUpdate,
  onDelete,
  onAddAfter
}: {
  note: string;
  index: number;
  onUpdate: (index: number, value: string) => void;
  onDelete: (index: number) => void;
  onAddAfter: (index: number) => void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(note);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setEditValue(note);
  }, [note]);

  useEffect(() => {
    if (isEditing && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.select();
    }
  }, [isEditing]);

  const handleSave = () => {
    const trimmed = editValue.trim();
    if (trimmed) {
      onUpdate(index, trimmed);
    } else {
      onDelete(index);
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      setEditValue(note);
      setIsEditing(false);
    } else if (e.key === 'Enter' && e.shiftKey) {
      e.preventDefault();
      handleSave();
      onAddAfter(index);
    }
  };

  if (isEditing) {
    return (
      <li className="flex items-start gap-3 text-gray-300">
        <span className="text-amber-400 mt-1 text-sm">•</span>
        <div className="flex-1">
          <textarea
            ref={textareaRef}
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={handleSave}
            onKeyDown={handleKeyDown}
            className="w-full bg-gray-700 text-gray-200 text-sm p-2 rounded border border-amber-500/50 focus:border-amber-400 focus:outline-none resize-none leading-relaxed"
            rows={Math.max(2, Math.ceil(editValue.length / 50))}
            placeholder="Enter note... (Shift+Enter for new note, Enter to save, Esc to cancel)"
          />
          <div className="text-xs text-gray-500 mt-1">
            Enter: save • Esc: cancel • Shift+Enter: save & add new • Empty: delete
          </div>
        </div>
      </li>
    );
  }

  return (
    <li
      className="flex items-start gap-3 text-gray-300 cursor-pointer hover:bg-gray-800/50 rounded p-1 -m-1 transition-colors"
      onClick={() => setIsEditing(true)}
    >
      <span className="text-amber-400 mt-1 text-sm">•</span>
      <span className="text-sm leading-relaxed">{note}</span>
    </li>
  );
}

export default function Presentation() {
  const [current, setCurrent] = useState(0);
  const [savedNotes, setSavedNotes] = useState<Record<number, string[]>>({});
  const [notesLoaded, setNotesLoaded] = useState(false);
  // Load saved notes on mount
  useEffect(() => {
    setSavedNotes(loadSavedNotes());
    setNotesLoaded(true);
  }, []);

  const next = () => setCurrent(c => Math.min(c + 1, slides.length - 1));
  const prev = () => setCurrent(c => Math.max(c - 1, 0));

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      // Don't navigate if we're in a textarea
      if ((e.target as HTMLElement)?.tagName === 'TEXTAREA') return;
      if (e.key === 'ArrowRight' || e.key === ' ') next();
      if (e.key === 'ArrowLeft') prev();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  const slide = slides[current];
  const currentNotes = getNotesForSlide(current, savedNotes);

  // Update a specific note
  const updateNote = (noteIndex: number, value: string) => {
    const newNotes = { ...savedNotes };
    const slideNotes = [...currentNotes];
    slideNotes[noteIndex] = value;
    newNotes[current] = slideNotes;
    setSavedNotes(newNotes);
    saveNotesToStorage(newNotes);
  };

  // Delete a note
  const deleteNote = (noteIndex: number) => {
    const newNotes = { ...savedNotes };
    const slideNotes = [...currentNotes];
    slideNotes.splice(noteIndex, 1);
    if (slideNotes.length === 0) {
      slideNotes.push('Add your notes here...');
    }
    newNotes[current] = slideNotes;
    setSavedNotes(newNotes);
    saveNotesToStorage(newNotes);
  };

  // Add a new note after a specific index
  const addNoteAfter = (noteIndex: number) => {
    const newNotes = { ...savedNotes };
    const slideNotes = [...currentNotes];
    slideNotes.splice(noteIndex + 1, 0, 'New note...');
    newNotes[current] = slideNotes;
    setSavedNotes(newNotes);
    saveNotesToStorage(newNotes);
  };

  // Add a new note at the end
  const addNote = () => {
    const newNotes = { ...savedNotes };
    const slideNotes = [...currentNotes];
    slideNotes.push('New note...');
    newNotes[current] = slideNotes;
    setSavedNotes(newNotes);
    saveNotesToStorage(newNotes);
  };

  // Reset notes for current slide to hardcoded defaults
  const resetNotes = () => {
    const newNotes = { ...savedNotes };
    delete newNotes[current];
    setSavedNotes(newNotes);
    saveNotesToStorage(newNotes);
  };
  
  return (
    <div className="min-h-screen bg-gray-800 text-white flex">
      {/* Main Slide Area - Fixed size for screen sharing */}
      <div data-slide-container className="w-[1280px] h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex flex-col">
        {/* Slide Header */}
        <div data-export-hide className="flex items-center justify-between p-4 text-sm text-gray-400 bg-gray-800/50">
          <span className="font-medium">Slide {current + 1} / {slides.length}</span>
          <div className="flex gap-1">
            {slides.map((_, i) => (
              <button
                key={i}
                onClick={() => setCurrent(i)}
                className={`w-3 h-3 rounded-full transition-colors ${i === current ? 'bg-blue-400' : 'bg-gray-600 hover:bg-gray-500'}`}
              />
            ))}
          </div>
        </div>
        
        {/* Slide Content - Fixed size */}
        <div className="flex-1 bg-gray-800/30 m-4 rounded-xl border border-gray-700/50 overflow-hidden">
          <div data-slide-content className="h-full p-8 overflow-y-auto">
            <h1 className="text-4xl font-bold text-white mb-2">{slide.title}</h1>
            <h2 className="text-xl text-blue-400 mb-8">{slide.subtitle}</h2>
            <div className="mt-6">
              {slide.content}
            </div>
          </div>
        </div>
        
        {/* Navigation Controls */}
        <div data-export-hide className="p-4 bg-gray-800/50 flex items-center justify-between">
          <button 
            onClick={prev} 
            disabled={current === 0} 
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 rounded disabled:opacity-30 hover:bg-gray-600 transition-colors"
          >
            <ChevronLeft size={18} /> Previous
          </button>
          <p className="text-gray-400 text-sm">← → arrows or spacebar to navigate</p>
          <button 
            onClick={next} 
            disabled={current === slides.length - 1} 
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 rounded disabled:opacity-30 hover:bg-blue-500 transition-colors"
          >
            Next <ChevronRight size={18} />
          </button>
        </div>
      </div>

      {/* Presenter Notes Panel - Fixed height, independent scroll */}
      <div data-export-hide className="w-[400px] h-screen bg-gray-900 border-l border-gray-700 flex flex-col overflow-hidden">
        <div className="p-4 bg-gray-800 border-b border-gray-700 shrink-0">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-amber-400 flex items-center gap-2">
              📝 Presenter Notes
            </h3>
            <div className="flex gap-2">
              <button
                onClick={addNote}
                className="text-xs px-2 py-1 bg-green-700 hover:bg-green-600 rounded transition-colors"
                title="Add new note"
              >
                + Add
              </button>
              {savedNotes[current] && (
                <button
                  onClick={resetNotes}
                  className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
                  title="Reset to default notes"
                >
                  Reset
                </button>
              )}
            </div>
          </div>
          {savedNotes[current] && (
            <div className="text-xs text-green-400 mt-1">✓ Custom notes saved</div>
          )}
        </div>

        <div className="flex-1 p-4 overflow-y-auto">
          <ul className="space-y-3">
            {currentNotes.map((note, index) => (
              <EditableNote
                key={`${current}-${index}`}
                note={note}
                index={index}
                onUpdate={updateNote}
                onDelete={deleteNote}
                onAddAfter={addNoteAfter}
              />
            ))}
          </ul>
        </div>
        
        {/* Quick slide navigation in notes panel */}
        <div className="p-4 bg-gray-800 border-t border-gray-700">
          <div className="text-xs text-gray-400 mb-2">Quick Navigation:</div>
          <div className="grid grid-cols-6 gap-1">
            {slides.map((_, i) => (
              <button 
                key={i} 
                onClick={() => setCurrent(i)} 
                className={`text-xs p-1 rounded ${i === current ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-400 hover:bg-gray-600'}`}
              >
                {i + 1}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
