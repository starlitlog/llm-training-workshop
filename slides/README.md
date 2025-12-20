# LLM Training Workshop - Slides

Interactive React-based presentation covering LLM training fundamentals, from theory to production deployment.

## 🎯 What's Covered

### **Core Theory** (Slides 1-10)
- **When to train custom models** vs using APIs
- **Cost analysis** - $100K/month APIs vs $50/month training
- **LLM evolution** - From perceptrons to Transformers  
- **The LLM ecosystem pyramid** - Where custom training fits

### **Practical Implementation** (Slides 11-20)
- **Critical hyperparameters** - Learning rate, batch size, LoRA config
- **Memory optimization** - Training on consumer GPUs
- **Training pipeline** - Data prep, training, evaluation
- **LoRA fine-tuning** - Memory-efficient approach

### **Production Deployment** (Slides 21-26)
- **Profiling & monitoring** - GPU utilization, memory tracking
- **Model evaluation** - ROUGE, BLEU, domain-specific metrics
- **Deployment workflows** - HuggingFace Hub, serving infrastructure
- **Learning resources** - Books, courses, communities to follow

## 🚀 Quick Start

### **Prerequisites**
- Node.js 16+ ([Download](https://nodejs.org/))
- Modern web browser

### **Run Presentation**
```bash
# Install dependencies
npm install

# Start development server  
npm run dev

# Open browser at http://localhost:5173
```

### **Navigation**
- **Arrow keys** or **click** to navigate slides
- **Presenter view** with speaker notes included
- **Mobile responsive** for tablets and phones

## 📊 Slide Structure

```
Slide 1:  Master LLM Training on Any Budget
Slide 2:  Why LLMs Now? The 80-Year Journey  
Slide 3:  Deep Learning Explosion (2010-2015)
Slide 4:  The Transformer Revolution (2017+)
...
Slide 25: Profiling & Performance Monitoring
Slide 26: Essential Learning Resources
```

## 🎨 Customization

### **Update Content**
Edit [`src/App.tsx`](./src/App.tsx) - all slide content is in the `slides` array.

### **Styling**
- Built with **Tailwind CSS** for easy customization
- Dark theme optimized for presentations
- Responsive design works on all devices

### **Add Slides**
```javascript
// Add to slides array in src/App.tsx
{
  title: "Your New Slide",
  subtitle: "Slide description",
  presenterNotes: [
    "Speaker note 1",
    "Speaker note 2"
  ],
  content: (
    <div>
      Your slide content here
    </div>
  )
}
```

## 🔧 Development

### **Build for Production**
```bash
npm run build
# Static files in dist/ folder
```

### **Preview Production Build**
```bash
npm run preview
```

### **Deployment Options**
- **GitHub Pages** - Push to gh-pages branch
- **Netlify** - Connect GitHub repo for auto-deployment
- **Vercel** - Import project for instant deployment
- **Self-hosted** - Serve dist/ folder with any web server

## 🎓 Using for Workshops

### **Presenter Tips**
- **Practice with speaker notes** - Press 'N' to show notes
- **Time management** - ~2-3 minutes per slide (60-90 min total)
- **Interactive elements** - Encourage questions on cost comparisons
- **Live coding** - Switch to hands-on project for demos

### **Workshop Flow**
1. **Slides 1-10**: Theory and motivation (30 min)
2. **Break**: Setup hands-on environment (15 min)  
3. **Slides 11-20**: Technical deep-dive (30 min)
4. **Hands-on**: Train first model (45 min)
5. **Slides 21-26**: Production topics (15 min)

### **Customizing for Your Audience**
- **Software engineers**: Emphasize production tooling and MLOps
- **Data scientists**: Focus on model architecture and evaluation  
- **Business stakeholders**: Highlight cost savings and business value
- **Students**: Add more theoretical background and resources

## 🔗 Integration with Hands-On

The slides reference the hands-on training project:

- **Slide 1**: Cost comparisons match hands-on training costs
- **Slide 12**: Hyperparameters match configuration files
- **Slide 25**: Profiling tools used in hands-on pipeline
- **Slide 26**: Resources continue the learning journey

### **Live Demo Integration**
```bash
# During presentation, switch to hands-on for live demos:
cd ../hands-on

# Quick training demo (5 minutes)
make train TRAIN_CONFIG=train_llama3b

# Show real results
make eval
```

## 📱 Mobile & Accessibility

- **Touch navigation** on mobile devices
- **Keyboard shortcuts** for accessibility
- **High contrast** text for visibility
- **Screen reader** compatible HTML structure

## 🤝 Contributing

To improve the slides:

1. **Fork the repository**
2. **Edit slides in src/App.tsx**  
3. **Test with `npm run dev`**
4. **Submit pull request**

**Common improvements:**
- Updated statistics and pricing
- New examples and case studies
- Better visualizations and diagrams
- Additional speaker notes

## 📄 License

MIT License - Use freely for workshops, training, and education.

---

## 🎬 Ready to Present?

**The slides provide the theory - the hands-on project provides the practice. Together, they create a complete learning experience for mastering LLM training.** 🚀

**Pro tip**: Practice the cost comparison section on Slide 1 - it's the most compelling part for business audiences!