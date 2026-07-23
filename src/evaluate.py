"""Evaluation script for Multi-Task Learning model."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm

from src.models.mtl_model import MultiTaskLearningModel
from src.data_processing.data_loader import FootballCommentaryDataset
from src.inference.commentator import AICommentator


class MTLEvaluator:
    """Evaluator for Multi-Task Learning model."""
    
    def __init__(self, model_path=None, device=None):
        """Initialize evaluator.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiTaskLearningModel()
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Action types mapping
        self.action_types = [
            'goal', 'pass', 'tackle', 'foul', 'corner', 'free_kick',
            'throw_in', 'substitution', 'yellow_card', 'red_card'
        ]
        
        # Sentiment mapping
        self.sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        print(f"Device: {self.device}")
    
    def prepare_labels(self, labels_dict):
        """Prepare labels for evaluation.
        
        Args:
            labels_dict: Dictionary with label information
            
        Returns:
            Tuple of (action_label, sentiment_label, importance_label)
        """
        action_type = labels_dict.get('action_type', 'pass')
        action_idx = self.action_types.index(action_type) if action_type in self.action_types else 0
        
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        sentiment = labels_dict.get('sentiment', 'neutral')
        sentiment_idx = sentiment_map.get(sentiment, 1)
        
        importance_score = float(labels_dict.get('importance_score', 5.0)) / 10.0
        
        return torch.tensor(action_idx), torch.tensor(sentiment_idx), torch.tensor(importance_score)
    
    def evaluate(self, test_data_path, batch_size=32):
        """Evaluate model on test data.
        
        Args:
            test_data_path: Path to test data
            batch_size: Batch size
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create dataset
        test_dataset = FootballCommentaryDataset(
            test_data_path,
            self.model.get_tokenizer(),
            max_length=512
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluation metrics
        all_action_preds = []
        all_action_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []
        all_importance_preds = []
        all_importance_labels = []
        
        total_loss = 0
        total_action_loss = 0
        total_sentiment_loss = 0
        total_importance_loss = 0
        
        print("\nEvaluating...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Prepare labels
                action_labels = []
                sentiment_labels = []
                importance_labels = []
                
                for label_dict in batch['labels']:
                    action_label, sentiment_label, importance_label = self.prepare_labels(label_dict)
                    action_labels.append(action_label)
                    sentiment_labels.append(sentiment_label)
                    importance_labels.append(importance_label)
                
                action_labels = torch.stack(action_labels).to(self.device)
                sentiment_labels = torch.stack(sentiment_labels).to(self.device)
                importance_labels = torch.stack(importance_labels).unsqueeze(1).to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Compute losses
                action_loss = self.ce_loss(outputs['action_logits'], action_labels)
                sentiment_loss = self.ce_loss(outputs['sentiment_logits'], sentiment_labels)
                importance_loss = self.mse_loss(outputs['importance_score'], importance_labels)
                
                total_loss += (action_loss.item() + sentiment_loss.item() + importance_loss.item())
                total_action_loss += action_loss.item()
                total_sentiment_loss += sentiment_loss.item()
                total_importance_loss += importance_loss.item()
                
                # Collect predictions
                action_preds = torch.argmax(outputs['action_logits'], dim=1).cpu().numpy()
                sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=1).cpu().numpy()
                importance_preds = outputs['importance_score'].cpu().numpy()
                
                all_action_preds.extend(action_preds)
                all_action_labels.extend(action_labels.cpu().numpy())
                all_sentiment_preds.extend(sentiment_preds)
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
                all_importance_preds.extend(importance_preds.flatten())
                all_importance_labels.extend(importance_labels.cpu().numpy().flatten())
        
        # Calculate metrics
        action_accuracy = accuracy_score(all_action_labels, all_action_preds)
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
        
        action_f1 = f1_score(all_action_labels, all_action_preds, average='weighted', zero_division=0)
        sentiment_f1 = f1_score(all_sentiment_labels, all_sentiment_preds, average='weighted', zero_division=0)
        
        # MAE for importance score
        importance_mae = np.mean(np.abs(np.array(all_importance_preds) - np.array(all_importance_labels)))
        
        results = {
            'action_accuracy': action_accuracy,
            'sentiment_accuracy': sentiment_accuracy,
            'action_f1': action_f1,
            'sentiment_f1': sentiment_f1,
            'importance_mae': importance_mae,
            'avg_loss': total_loss / len(test_loader),
            'action_loss': total_action_loss / len(test_loader),
            'sentiment_loss': total_sentiment_loss / len(test_loader),
            'importance_loss': total_importance_loss / len(test_loader)
        }
        
        return results
    
    def print_results(self, results):
        """Print evaluation results.
        
        Args:
            results: Dictionary with evaluation metrics
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Action Recognition Accuracy: {results['action_accuracy']:.4f}")
        print(f"Action Recognition F1-Score: {results['action_f1']:.4f}")
        print(f"\nSentiment Analysis Accuracy: {results['sentiment_accuracy']:.4f}")
        print(f"Sentiment Analysis F1-Score: {results['sentiment_f1']:.4f}")
        print(f"\nImportance Score MAE: {results['importance_mae']:.4f}")
        print(f"\nTotal Loss: {results['avg_loss']:.4f}")
        print(f"  - Action Loss: {results['action_loss']:.4f}")
        print(f"  - Sentiment Loss: {results['sentiment_loss']:.4f}")
        print(f"  - Importance Loss: {results['importance_loss']:.4f}")
        print("="*50)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate MTL AI Commentator')
    parser.add_argument('--model', type=str, default='models/mtl_model_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default='data/processed/test.json',
                       help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MTLEvaluator(args.model)
    
    # Evaluate
    results = evaluator.evaluate(args.test_data, args.batch_size)
    
    # Print results
    evaluator.print_results(results)


if __name__ == '__main__':
    main()
