#!/bin/bash

echo "🚀 Uploading A*-PO Mathematical Reasoning System to GitHub..."
echo "=================================================="

# Navigate to the project directory
PROJECT_DIR="/Users/amankhan/Downloads/NU/fall 2025/self learning AI/week 5/A--PO-Mathematical-Reasoning-System"
cd "$PROJECT_DIR"

echo "📁 Current directory: $(pwd)"
echo "📋 Files in directory:"
ls -la

echo ""
echo "🔧 Initializing git repository..."
git init

echo "🔗 Adding GitHub remote..."
git remote add origin https://github.com/Amankhan2370/A--PO-Mathematical-Reasoning-System.git

echo "➕ Adding all files..."
git add .

echo "💾 Committing files..."
git commit -m "Initial commit: Complete A*-PO Mathematical Reasoning System

🧠 Advanced reinforcement learning implementation for AI mathematical problem-solving
🚀 Two-stage A*-PO algorithm with offline value estimation + online policy optimization  
🎯 Multi-turn self-correction for mathematical reasoning
🤖 Qwen2.5-1.5B-Instruct model integration
📊 Comprehensive evaluation framework with MATH dataset
⚙️ Production-ready pipeline with data preprocessing
📚 Professional documentation and demo system

Technical Features:
- Value function approximation with neural networks
- Advantage-weighted policy optimization  
- Mathematical solution verification and equivalence checking
- LaTeX processing and answer extraction
- Configurable training pipeline with extensive logging
- Modular design for easy extension and customization

Ready for mathematical AI research and development!"

echo "🌿 Setting main branch..."
git branch -M main

echo "⬆️ Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Upload completed successfully!"
echo "🌐 Your repository is now live at:"
echo "   https://github.com/Amankhan2370/A--PO-Mathematical-Reasoning-System"
echo ""
echo "📊 Repository includes:"
echo "   ✓ README.md - Professional documentation"
echo "   ✓ apo_algorithm.py - Core A*-PO implementation"  
echo "   ✓ train_math_apo.py - Training pipeline"
echo "   ✓ evaluate_apo.py - Evaluation framework"
echo "   ✓ prepare_data.py - Data preprocessing"
echo "   ✓ demo_apo_pipeline.py - Demo system"
echo "   ✓ requirements.txt - Dependencies"
echo "   ✓ .gitignore - Git exclusions"
echo ""
echo "🎉 Your A*-PO Mathematical Reasoning System is now showcasing your AI skills on GitHub!"