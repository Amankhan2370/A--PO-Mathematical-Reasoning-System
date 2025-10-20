#!/bin/bash

# Upload A*-PO project to GitHub
echo "Starting upload to GitHub..."

# Navigate to project directory
cd "/Users/amankhan/Downloads/NU/fall 2025/self learning AI/week 5/A--PO-Mathematical-Reasoning-System"

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git remote add origin https://github.com/Amankhan2370/A--PO-Mathematical-Reasoning-System.git
fi

# Configure git user (if needed)
git config user.name "Amankhan2370"
git config user.email "amankhan2370@users.noreply.github.com"

# Add all files
echo "Adding files to git..."
git add .

# Commit files
echo "Committing files..."
git commit -m "Initial commit: Complete A*-PO Mathematical Reasoning System

Features:
- Advanced A*-PO reinforcement learning algorithm implementation
- Two-stage training: offline value estimation + online policy optimization  
- Multi-turn self-correction for mathematical reasoning
- Qwen2.5-1.5B-Instruct model integration
- Comprehensive evaluation framework with MATH dataset
- Production-ready pipeline with data preprocessing
- Professional documentation and demo system

Technical Highlights:
- Value function approximation with neural networks
- Advantage-weighted policy optimization
- Mathematical solution verification and equivalence checking
- LaTeX processing and answer extraction
- Configurable training pipeline with extensive logging
- Modular design for easy extension and customization"

# Create main branch and push
echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo "Upload complete!"
echo "Repository available at: https://github.com/Amankhan2370/A--PO-Mathematical-Reasoning-System"