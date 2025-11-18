#!/usr/bin/env python3
"""
Setup verification script for BERT Document Classifier.
Checks if all required files and dependencies are in place.
"""

import sys
from pathlib import Path
import importlib.util

def check_file(filepath, description):
    """Check if a file exists."""
    if filepath.exists():
        print(f"[OK] {description}: Found")
        return True
    else:
        print(f"[MISSING] {description}: {filepath}")
        return False

def check_package(package_name):
    """Check if a Python package is installed."""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        print(f"[OK] Package '{package_name}': Installed")
        return True
    else:
        print(f"[MISSING] Package '{package_name}': Not installed")
        return False

def main():
    print("=" * 60)
    print("BERT Document Classifier - Setup Verification")
    print("=" * 60)
    print()
    
    base_dir = Path(__file__).parent
    all_good = True
    
    print("Checking Model Files:")
    print("-" * 60)
    model_dir = base_dir / "saved_model"
    
    all_good &= check_file(model_dir / "model.pt", "Model weights")
    all_good &= check_file(model_dir / "label_encoder.joblib", "Label encoder")
    print()
    
    print("Checking Required Packages:")
    print("-" * 60)
    required_packages = [
        'torch',
        'transformers',
        'fastapi',
        'uvicorn',
        'joblib',
        'fitz',
    ]
    
    for package in required_packages:
        all_good &= check_package(package)
    print()
    
    print("Checking Code Files:")
    print("-" * 60)
    all_good &= check_file(base_dir / "model" / "bert_classifier.py", "Model definition")
    all_good &= check_file(base_dir / "model" / "inference.py", "Inference module")
    all_good &= check_file(base_dir / "api" / "app.py", "API application")
    all_good &= check_file(base_dir / "requirements.txt", "Requirements file")
    print()
    
    print("=" * 60)
    if all_good:
        print("All checks passed. Ready to run.")
        print("\nTo start the API server:")
        print("  uvicorn api.app:app --reload")
        sys.exit(0)
    else:
        print("Some checks failed. Please address the issues above.")
        print("\nIf model files are missing:")
        print("  Run: experiment_01_document_classifier.ipynb")
        print("\nIf packages are missing:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
