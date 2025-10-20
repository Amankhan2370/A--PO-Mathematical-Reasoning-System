#!/bin/bash

# A*-PO Project Completion Check
echo "🔍 A*-PO PROJECT STATUS CHECK"
echo "=============================="

# Check if we're in the right directory
if [ ! -f "apo_algorithm.py" ]; then
    echo "❌ Not in the correct directory. Please navigate to the math project folder."
    exit 1
fi

echo "📁 Checking core files..."

# Check core implementation files
files=("prepare_data.py" "apo_algorithm.py" "train_math_apo.py" "evaluate_apo.py" "demo_apo_pipeline.py" "requirements.txt" "README.md")
missing_files=0

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file" 2>/dev/null || echo "0")
        echo "  ✅ $file ($size bytes)"
    else
        echo "  ❌ $file (missing)"
        ((missing_files++))
    fi
done

echo ""
echo "📊 Checking data files..."

# Check data files
data_files=("math_prompts_responses.jsonl" "math_subset.jsonl" "math_eval_500.jsonl")
missing_data=0

for file in "${data_files[@]}"; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file" 2>/dev/null || echo "0")
        echo "  ✅ $file ($lines lines)"
    else
        echo "  ❌ $file (missing)"
        ((missing_data++))
    fi
done

echo ""
echo "🤖 Checking model artifacts..."

if [ -d "./apo_trained_model" ]; then
    echo "  ✅ apo_trained_model/ (trained model present)"
    model_trained=1
else
    echo "  ❌ apo_trained_model/ (no trained model)"
    model_trained=0
fi

echo ""
echo "📈 PROJECT SUMMARY:"
echo "==================="

total_files=${#files[@]}
core_complete=$((total_files - missing_files))
echo "Core Implementation: $core_complete/$total_files files"

total_data=${#data_files[@]}
data_complete=$((total_data - missing_data))
echo "Data Preparation: $data_complete/$total_data datasets"

echo "Model Training: $([ $model_trained -eq 1 ] && echo "✅ Complete" || echo "❌ Not done")"

# Calculate completion percentage
core_pct=$((core_complete * 60 / total_files))
data_pct=$((data_complete * 25 / total_data))
model_pct=$((model_trained * 15))
total_pct=$((core_pct + data_pct + model_pct))

echo ""
echo "🎯 OVERALL COMPLETION: $total_pct%"

if [ $total_pct -ge 85 ]; then
    echo "🎉 EXCELLENT! Your A*-PO project is research-quality!"
elif [ $total_pct -ge 70 ]; then
    echo "👍 GREAT! Nearly complete - just run training!"
elif [ $total_pct -ge 50 ]; then
    echo "👌 GOOD! Most components ready."
else
    echo "🚧 KEEP GOING! You're building something impressive."
fi

echo ""
echo "💡 NEXT STEPS:"
if [ $missing_files -eq 0 ]; then
    echo "  ✅ Implementation complete!"
else
    echo "  📝 Complete missing implementation files"
fi

if [ $missing_data -lt $total_data ]; then
    echo "  ✅ Data preparation done!"
else
    echo "  📝 Run: python prepare_data.py"
fi

if [ $model_trained -eq 0 ]; then
    echo "  📝 Run training: python train_math_apo.py"
    echo "  📝 Then evaluate: python evaluate_apo.py"
else
    echo "  ✅ Training complete! Run: python evaluate_apo.py"
fi

echo ""
echo "=============================="
echo "Status check complete!"

# Save status to file
cat > project_status.txt << EOF
A*-PO Project Status Report
===========================
Date: $(date)
Core Implementation: $core_complete/$total_files files
Data Preparation: $data_complete/$total_data datasets  
Model Training: $([ $model_trained -eq 1 ] && echo "Complete" || echo "Not done")
Overall Completion: $total_pct%

$([ $total_pct -ge 85 ] && echo "🎉 EXCELLENT! Research-quality implementation!" || echo "Keep going - you're doing great!")
EOF

echo "📄 Status saved to project_status.txt"
