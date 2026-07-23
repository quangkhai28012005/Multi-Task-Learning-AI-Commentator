# Multi-Task Learning AI Commentator - Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip or conda
- (Optional) CUDA 11.0+ for GPU support

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/quangkhai28012005/Multi-Task-Learning-AI-Commentator.git
cd Multi-Task-Learning-AI-Commentator
```

### 2. Create Virtual Environment

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment

```bash
python setup_env.py
```

## Quick Start

### Run Quick Test

```bash
python main.py --test
```

This will verify that all components are working correctly.

### Run Demo

```bash
# Batch mode (analyze multiple events)
python main.py --demo

# Single event mode
python main.py --demo --demo-mode single
```

### Prepare Training Data

```bash
python main.py --prepare-data
```

This generates training, validation, and test datasets.

### Train Model

```bash
python main.py --train --train-config config/training_config.yaml
```

To train on GPU:
```bash
python main.py --train --device cuda
```

### Evaluate Model

```bash
python main.py --evaluate --model models/mtl_model_best.pt
```

### Run Unit Tests

```bash
pytest tests/
```

## Project Structure

```
.
├── main.py                          # Main entry point
├── setup_env.py                     # Environment setup
├── requirements.txt                 # Dependencies
├── README.md                        # Project documentation
├── INSTALL.md                       # This file
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── mtl_model.py            # Multi-task learning model
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Data loading utilities
│   │   └── preprocessing.py        # Text preprocessing
│   ├── inference/
│   │   ├── __init__.py
│   │   └── commentator.py          # AI commentator engine
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── losses.py               # Custom loss functions
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py              # Utility functions
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   └── demo.py                     # Demo script
│
├── data/
│   ├── raw/                        # Raw data directory
│   ├── processed/                  # Processed data directory
│   └── samples/                    # Sample data
│
├── models/                         # Trained model checkpoints
│   └── checkpoints/               # Training checkpoints
│
├── notebooks/
│   └── Demo.ipynb                 # Jupyter demo notebook
│
├── scripts/
│   ├── prepare_data.py            # Data preparation script
│   └── quick_test.py              # Quick test script
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py              # Model tests
│   └── test_inference.py          # Inference tests
│
and config/
│   └── training_config.yaml       # Training configuration
```

## Troubleshooting

### Issue: CUDA not available

If you want to use CPU instead:
```bash
python main.py --test --device cpu
```

### Issue: Missing dependencies

Re-install all dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Module not found errors

Ensure you're in the correct directory and the virtual environment is activated:
```bash
cd Multi-Task-Learning-AI-Commentator
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Next Steps

1. **Prepare Data**: Run `python main.py --prepare-data` to generate training datasets
2. **Train Model**: Run `python main.py --train` to train the model
3. **Evaluate**: Run `python main.py --evaluate` to test the trained model
4. **Demo**: Run `python main.py --demo` to see live analysis

## Support

If you encounter any issues, please open an issue on the GitHub repository.
