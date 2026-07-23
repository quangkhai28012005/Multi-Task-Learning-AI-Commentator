"""Configuration and setup utilities."""

import os
from pathlib import Path


def setup_environment():
    """Setup project environment."""
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed',
        'data/samples',
        'models',
        'models/checkpoints',
        'logs',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Environment setup complete")


def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        'torch',
        'transformers',
        'pandas',
        'numpy',
        'scikit-learn',
        'tqdm',
        'pyyaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    print("✓ All requirements satisfied")
    return True


if __name__ == '__main__':
    setup_environment()
    check_requirements()
