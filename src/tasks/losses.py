"""Custom loss functions for multi-task learning."""

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining different task losses."""
    
    def __init__(self, task_weights=None):
        """Initialize multi-task loss.
        
        Args:
            task_weights: Dictionary with weights for each task
        """
        super().__init__()
        
        if task_weights is None:
            task_weights = {
                'action_recognition': 1.0,
                'sentiment_analysis': 1.0,
                'event_evaluation': 0.5
            }
        
        self.task_weights = task_weights
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, action_logits, sentiment_logits, importance_score,
                action_labels, sentiment_labels, importance_labels):
        """Calculate multi-task loss.
        
        Args:
            action_logits: Action classification logits
            sentiment_logits: Sentiment classification logits
            importance_score: Importance regression output
            action_labels: True action labels
            sentiment_labels: True sentiment labels
            importance_labels: True importance labels
            
        Returns:
            Total weighted loss
        """
        action_loss = self.ce_loss(action_logits, action_labels)
        sentiment_loss = self.ce_loss(sentiment_logits, sentiment_labels)
        importance_loss = self.mse_loss(importance_score, importance_labels)
        
        total_loss = (
            self.task_weights['action_recognition'] * action_loss +
            self.task_weights['sentiment_analysis'] * sentiment_loss +
            self.task_weights['event_evaluation'] * importance_loss
        )
        
        return total_loss, {
            'action_loss': action_loss.item(),
            'sentiment_loss': sentiment_loss.item(),
            'importance_loss': importance_loss.item()
        }
