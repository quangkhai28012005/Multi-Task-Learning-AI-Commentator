"""Training script for Multi-Task Learning model."""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
from pathlib import Path

from src.models.mtl_model import MultiTaskLearningModel
from src.data_processing.data_loader import FootballCommentaryDataset
from src.utils.helpers import load_config, set_seed, create_output_dir, load_json


class MTLTrainer:
    """Trainer for Multi-Task Learning model."""
    
    def __init__(self, config_path, device=None):
        """Initialize trainer.
        
        Args:
            config_path: Path to training config
            device: Device to use
        """
        self.config = load_config(config_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed
        set_seed(self.config['training']['seed'])
        
        # Create output directories
        create_output_dir(self.config['output_dir'])
        create_output_dir(self.config['checkpoint_dir'])
        create_output_dir(self.config['log_dir'])
        
        # Initialize model
        self.model = MultiTaskLearningModel()
        self.model.to(self.device)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def prepare_labels(self, labels_dict, tokenizer):
        """Prepare labels for training.
        
        Args:
            labels_dict: Dictionary with label information
            tokenizer: Tokenizer (not used but kept for future extension)
            
        Returns:
            Tuple of (action_label, sentiment_label, importance_label)
        """
        # Action type mapping
        action_types = [
            'goal', 'pass', 'tackle', 'foul', 'corner', 'free_kick',
            'throw_in', 'substitution', 'yellow_card', 'red_card'
        ]
        
        # Sentiment mapping
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        action_type = labels_dict.get('action_type', 'pass')
        action_idx = action_types.index(action_type) if action_type in action_types else 0
        
        sentiment = labels_dict.get('sentiment', 'neutral')
        sentiment_idx = sentiment_map.get(sentiment, 1)
        
        importance_score = float(labels_dict.get('importance_score', 5.0)) / 10.0  # Normalize to [0, 1]
        
        return torch.tensor(action_idx), torch.tensor(sentiment_idx), torch.tensor(importance_score)
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        
        total_action_loss = 0
        total_sentiment_loss = 0
        total_importance_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Prepare labels
            action_labels = []
            sentiment_labels = []
            importance_labels = []
            
            for label_dict in batch['labels']:
                action_label, sentiment_label, importance_label = self.prepare_labels(
                    label_dict, self.model.get_tokenizer()
                )
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
            
            # Weighted loss
            loss_weights = self.config['loss_weights']
            total_loss = (
                loss_weights['action_recognition'] * action_loss +
                loss_weights['sentiment_analysis'] * sentiment_loss +
                loss_weights['event_evaluation'] * importance_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            optimizer.step()
            scheduler.step()
            
            # Track losses
            total_action_loss += action_loss.item()
            total_sentiment_loss += sentiment_loss.item()
            total_importance_loss += importance_loss.item()
            
            progress_bar.set_postfix({
                'action_loss': action_loss.item(),
                'sentiment_loss': sentiment_loss.item(),
                'importance_loss': importance_loss.item()
            })
        
        return {
            'action_loss': total_action_loss / len(train_loader),
            'sentiment_loss': total_sentiment_loss / len(train_loader),
            'importance_loss': total_importance_loss / len(train_loader)
        }
    
    def validate(self, val_loader):
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with loss values
        """
        self.model.eval()
        
        total_action_loss = 0
        total_sentiment_loss = 0
        total_importance_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Prepare labels
                action_labels = []
                sentiment_labels = []
                importance_labels = []
                
                for label_dict in batch['labels']:
                    action_label, sentiment_label, importance_label = self.prepare_labels(
                        label_dict, self.model.get_tokenizer()
                    )
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
                
                total_action_loss += action_loss.item()
                total_sentiment_loss += sentiment_loss.item()
                total_importance_loss += importance_loss.item()
        
        return {
            'action_loss': total_action_loss / len(val_loader),
            'sentiment_loss': total_sentiment_loss / len(val_loader),
            'importance_loss': total_importance_loss / len(val_loader)
        }
    
    def train(self, train_data_path, val_data_path):
        """Train the model.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
        """
        # Create datasets
        train_dataset = FootballCommentaryDataset(
            train_data_path,
            self.model.get_tokenizer(),
            max_length=self.config['data']['max_length']
        )
        val_dataset = FootballCommentaryDataset(
            val_data_path,
            self.model.get_tokenizer(),
            max_length=self.config['data']['max_length']
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        total_steps = len(train_loader) * self.config['training']['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train': [], 'val': []}
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            # Train
            train_losses = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Calculate total losses
            train_total_loss = (
                self.config['loss_weights']['action_recognition'] * train_losses['action_loss'] +
                self.config['loss_weights']['sentiment_analysis'] * train_losses['sentiment_loss'] +
                self.config['loss_weights']['event_evaluation'] * train_losses['importance_loss']
            )
            val_total_loss = (
                self.config['loss_weights']['action_recognition'] * val_losses['action_loss'] +
                self.config['loss_weights']['sentiment_analysis'] * val_losses['sentiment_loss'] +
                self.config['loss_weights']['event_evaluation'] * val_losses['importance_loss']
            )
            
            history['train'].append(train_losses)
            history['val'].append(val_losses)
            
            print(f"Train Loss: {train_total_loss:.4f}")
            print(f"  - Action: {train_losses['action_loss']:.4f}")
            print(f"  - Sentiment: {train_losses['sentiment_loss']:.4f}")
            print(f"  - Importance: {train_losses['importance_loss']:.4f}")
            
            print(f"Val Loss: {val_total_loss:.4f}")
            print(f"  - Action: {val_losses['action_loss']:.4f}")
            print(f"  - Sentiment: {val_losses['sentiment_loss']:.4f}")
            print(f"  - Importance: {val_losses['importance_loss']:.4f}")
            
            # Save best model
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                self.save_model(f"{self.config['output_dir']}/mtl_model_best.pt")
                print("✓ Best model saved!")
            
            # Save checkpoint
            self.save_model(f"{self.config['checkpoint_dir']}/mtl_model_epoch_{epoch + 1}.pt")
        
        # Save training history
        with open(f"{self.config['log_dir']}/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✓ Training completed! Best model saved at {self.config['output_dir']}/mtl_model_best.pt")
    
    def save_model(self, path):
        """Save model checkpoint.
        
        Args:
            path: Path to save model
        """
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model checkpoint.
        
        Args:
            path: Path to load model
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MTL AI Commentator')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to config file')
    parser.add_argument('--train_data', type=str, default='data/processed/train.json',
                       help='Path to training data')
    parser.add_argument('--val_data', type=str, default='data/processed/val.json',
                       help='Path to validation data')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MTLTrainer(args.config)
    
    # Train
    trainer.train(args.train_data, args.val_data)


if __name__ == '__main__':
    main()
