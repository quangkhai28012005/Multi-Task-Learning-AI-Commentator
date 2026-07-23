"""Main entry point for running the AI Commentator system."""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Multi-Task Learning AI Commentator for Football Match Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test
  python main.py --test
  
  # Run demo analysis
  python main.py --demo
  
  # Prepare training data
  python main.py --prepare-data
  
  # Train model
  python main.py --train
  
  # Evaluate model
  python main.py --evaluate
        """
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Run quick test')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with sample data')
    parser.add_argument('--demo-mode', choices=['batch', 'single'],
                       default='batch', help='Demo mode')
    parser.add_argument('--prepare-data', action='store_true',
                       help='Prepare training data')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--train-config', type=str,
                       default='config/training_config.yaml',
                       help='Path to training config')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the model')
    parser.add_argument('--model', type=str,
                       default='models/mtl_model_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', choices=['cpu', 'cuda'],
                       help='Device to use (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    if not any([args.test, args.demo, args.prepare_data, args.train, args.evaluate]):
        print("No action specified. Use --help for usage information.")
        parser.print_help()
        return
    
    # Quick test
    if args.test:
        print("Running quick test...")
        from scripts.quick_test import quick_test
        quick_test()
    
    # Demo
    if args.demo:
        print("Running demo...")
        from src.demo import demo_basic, demo_single_event
        if args.demo_mode == 'batch':
            demo_basic()
        else:
            demo_single_event()
    
    # Prepare data
    if args.prepare_data:
        print("Preparing training data...")
        from scripts.prepare_data import prepare_data
        prepare_data()
    
    # Train
    if args.train:
        print("Training model...")
        from src.train import MTLTrainer
        trainer = MTLTrainer(args.train_config, device=args.device)
        trainer.train('data/processed/train.json', 'data/processed/val.json')
    
    # Evaluate
    if args.evaluate:
        print("Evaluating model...")
        from src.evaluate import MTLEvaluator
        evaluator = MTLEvaluator(args.model, device=args.device)
        results = evaluator.evaluate('data/processed/test.json')
        evaluator.print_results(results)


if __name__ == '__main__':
    main()
