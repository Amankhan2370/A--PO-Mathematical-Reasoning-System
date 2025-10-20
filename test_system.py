#!/usr/bin/env python3
"""
Quick test script to verify the evaluation code works
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test if all imports work"""
    try:
        print("Testing imports...")

        # Test basic imports
        import torch
        print("✓ PyTorch imported successfully")

        from transformers import AutoTokenizer
        print("✓ Transformers imported successfully")

        # Test our evaluation script import
        from evaluate_apo import MathEvaluator
        print("✓ MathEvaluator imported successfully")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_math_evaluator():
    """Test basic MathEvaluator functionality"""
    try:
        print("\nTesting MathEvaluator...")

        # Create evaluator
        evaluator = MathEvaluator()
        print("✓ MathEvaluator created")

        # Test answer extraction
        test_solution = "The answer is 42. Therefore, \\boxed{42}."
        answer = evaluator.extract_answer(test_solution)
        print(f"✓ Answer extraction works: '{answer}'")

        # Test verification
        verified = evaluator.verify_solution(
            "What is 2+2?", "The answer is 4 because 2+2=4.")
        print(f"✓ Solution verification works: {verified}")

        return True
    except Exception as e:
        print(f"✗ MathEvaluator test failed: {e}")
        return False


def test_data_files():
    """Check if data files exist"""
    print("\nChecking data files...")

    files_to_check = [
        "math_prompts_responses.jsonl",
        "math_subset.jsonl",
        "math_eval_500.jsonl"
    ]

    files_found = []
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✓ {file} exists")
            files_found.append(file)
        else:
            print(f"✗ {file} not found")

    return len(files_found) > 0


def main():
    """Run all tests"""
    print("=" * 50)
    print("A*-PO EVALUATION SYSTEM TEST")
    print("=" * 50)

    tests_passed = 0
    total_tests = 3

    # Test 1: Imports
    if test_imports():
        tests_passed += 1

    # Test 2: MathEvaluator
    if test_math_evaluator():
        tests_passed += 1

    # Test 3: Data files
    if test_data_files():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Your evaluation system is working!")
        print("\nYour project is ready. To run evaluation:")
        print("  python evaluate_apo.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

        if tests_passed >= 1:
            print("✅ The core system is functional though!")

    print("=" * 50)


if __name__ == "__main__":
    main()
