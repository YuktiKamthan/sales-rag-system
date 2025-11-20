#!/usr/bin/env python3
"""
Quick Start Verification Script
Checks if everything is set up correctly
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_files():
    """Check if all required files exist."""
    print("\n📁 Checking project structure...")
    required_files = [
        'requirements.txt',
        'config/config.yaml',
        'src/data_loader.py',
        'src/embeddings.py',
        'src/vector_store.py',
        'src/rag_pipeline.py',
        'src/predictor.py',
        'src/telemetry.py',
        'app.py'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} (missing)")
            all_exist = False
    
    return all_exist

def check_data():
    """Check if data file exists."""
    print("\n📊 Checking data file...")
    if Path('data/sales_data.xlsx').exists():
        print("   ✓ data/sales_data.xlsx")
        return True
    else:
        print("   ✗ data/sales_data.xlsx (missing)")
        print("   → Place your sales data in data/sales_data.xlsx")
        return False

def check_env():
    """Check if .env file exists."""
    print("\n🔑 Checking environment setup...")
    if Path('.env').exists():
        print("   ✓ .env file exists")
        
        # Check if API key is set
        with open('.env', 'r') as f:
            content = f.read()
            if 'GEMINI_API_KEY' in content and 'your_' not in content:
                print("   ✓ GEMINI_API_KEY appears to be set")
                return True
            else:
                print("   ⚠️  GEMINI_API_KEY not configured")
                print("   → Edit .env and add your Gemini API key")
                return False
    else:
        print("   ✗ .env file not found")
        print("   → Copy .env.example to .env and add your API key")
        return False

def check_packages():
    """Check if key packages can be imported."""
    print("\n📦 Checking installed packages...")
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'chromadb': 'chromadb',
        'sentence_transformers': 'sentence-transformers',
        'google.generativeai': 'google-generativeai',
        'prophet': 'prophet',
        'opentelemetry': 'opentelemetry-api'
    }
    
    all_installed = True
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} (not installed)")
            all_installed = False
    
    if not all_installed:
        print("\n   → Run: pip install -r requirements.txt")
    
    return all_installed

def main():
    """Run all checks."""
    print("=" * 60)
    print("🚀 Sales RAG System - Setup Verification")
    print("=" * 60)
    
    checks = {
        'Python Version': check_python_version(),
        'Project Files': check_files(),
        'Data File': check_data(),
        'Environment': check_env(),
        'Packages': check_packages()
    }
    
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    
    for name, status in checks.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{name:20} {status_str}")
    
    if all(checks.values()):
        print("\n🎉 All checks passed! You're ready to go!")
        print("\nNext steps:")
        print("1. Run: python app.py rag")
        print("2. Or:  python app.py predict")
        print("3. Or:  python app.py stats")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor help, see CURSOR_GUIDE.md")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
