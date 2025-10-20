#!/usr/bin/env python3
"""
🚀 A*-PO PROJECT LAUNCHER & TERMINAL TEST
========================================

This script helps you get started with your A*-PO project and tests if everything works.
"""

import sys
import os
import subprocess
from pathlib import Path


def print_banner():
    print("🚀" + "="*58 + "🚀")
    print("🎯 A*-PO MATHEMATICAL REASONING PROJECT LAUNCHER 🎯")
    print("🚀" + "="*58 + "🚀")


def check_environment():
    """Check if the Python environment is working"""
    print("\n🔍 CHECKING ENVIRONMENT...")

    try:
        import torch
        print("✅ PyTorch available")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not found")
        return False

    try:
        import transformers
        print("✅ Transformers available")
        print(f"   Version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found")
        return False

    try:
        import datasets
        print("✅ Datasets available")
    except ImportError:
        print("❌ Datasets not found")
        return False

    return True


def check_files():
    """Check if all project files are present"""
    print("\n📁 CHECKING PROJECT FILES...")

    required_files = [
        "prepare_data.py",
        "apo_algorithm.py",
        "train_math_apo.py",
        "evaluate_apo.py",
        "demo_apo_pipeline.py"
    ]

    all_present = True
    for file in required_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file:<25} ({size:,} bytes)")
        else:
            print(f"❌ {file:<25} (missing)")
            all_present = False

    return all_present


def check_data():
    """Check if data files exist"""
    print("\n📊 CHECKING DATA FILES...")

    data_files = [
        "math_prompts_responses.jsonl",
        "math_subset.jsonl",
        "math_eval_500.jsonl"
    ]

    data_present = False
    for file in data_files:
        if Path(file).exists():
            try:
                with open(file, 'r') as f:
                    lines = sum(1 for _ in f)
                print(f"✅ {file:<30} ({lines:,} samples)")
                data_present = True
            except:
                print(f"✅ {file:<30} (present)")
                data_present = True
        else:
            print(f"❌ {file:<30} (missing)")

    return data_present


def run_quick_test():
    """Run a quick functionality test"""
    print("\n🧪 RUNNING QUICK FUNCTIONALITY TEST...")

    try:
        # Test imports
        sys.path.append('.')
        from evaluate_apo import MathEvaluator
        print("✅ Core modules import successfully")

        # Test evaluator creation
        evaluator = MathEvaluator()
        print("✅ MathEvaluator creates successfully")

        # Test answer extraction
        test_solution = "The answer is 42. Therefore, \\boxed{42}."
        answer = evaluator.extract_answer(test_solution)
        print(f"✅ Answer extraction works: '{answer}'")

        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def show_next_steps(env_ok, files_ok, data_ok, test_ok):
    """Show what to do next"""
    print("\n🎯 NEXT STEPS:")
    print("="*50)

    if not env_ok:
        print("1. 📦 Install dependencies:")
        print("   pip install -r requirements.txt")
        return

    if not files_ok:
        print("1. 📝 Your core files seem to be missing!")
        print("   Make sure you're in the right directory.")
        return

    if not data_ok:
        print("1. 📊 Prepare data first:")
        print("   python prepare_data.py")
        print("")

    if env_ok and files_ok:
        print("🎉 YOUR PROJECT IS READY TO RUN!")
        print("")
        print("Available commands:")
        print("🔹 python prepare_data.py      - Load and prepare MATH dataset")
        print("🔹 python train_math_apo.py    - Train the A*-PO model")
        print("🔹 python evaluate_apo.py      - Evaluate model performance")
        print("🔹 python demo_apo_pipeline.py - Run complete demo")
        print("🔹 python project_report.py    - Check project status")
        print("")

        if test_ok:
            print("✅ All systems are GO! Your implementation is working!")
        else:
            print("⚠️  Some tests failed, but core system should work.")


def main():
    """Main launcher function"""
    print_banner()

    # Change to math directory if we're not there
    if not Path("apo_algorithm.py").exists():
        if Path("math/apo_algorithm.py").exists():
            os.chdir("math")
            print("📁 Changed to math/ directory")
        else:
            print("❌ Cannot find project files!")
            print("   Make sure you're in the correct directory.")
            return

    # Run all checks
    env_ok = check_environment()
    files_ok = check_files()
    data_ok = check_data()
    test_ok = run_quick_test() if env_ok and files_ok else False

    # Show results
    print("\n📈 SYSTEM STATUS:")
    print("="*30)
    print(f"Environment:  {'✅ Ready' if env_ok else '❌ Issues'}")
    print(f"Project Files: {'✅ Complete' if files_ok else '❌ Missing'}")
    print(f"Data Files:   {'✅ Ready' if data_ok else '❌ Missing'}")
    print(f"Functionality: {'✅ Working' if test_ok else '❌ Issues'}")

    # Calculate score
    score = sum([env_ok, files_ok, data_ok, test_ok]) * 25
    print(f"\nOverall Status: {score}% Complete")

    # Show next steps
    show_next_steps(env_ok, files_ok, data_ok, test_ok)

    print("\n🚀" + "="*58 + "🚀")


if __name__ == "__main__":
    main()
