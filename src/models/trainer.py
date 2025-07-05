"""
A-TCN Trainer for Vietnamese Accent Restoration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
try:
    from .atcn import ATCN
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from atcn import ATCN


class ATCNTrainer:
    """
    Trainer class for A-TCN model
    Handles training and validation steps, loss computation, and metrics
    """
    
    def __init__(self, model: ATCN, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute character-level cross-entropy loss
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            loss: scalar tensor
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten for cross-entropy loss
        logits_flat = logits.reshape(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        targets_flat = targets.reshape(-1)  # [batch_size * seq_len]
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask_flat = attention_mask.reshape(-1)  # [batch_size * seq_len]
            loss = loss * mask_flat  # Zero out padded positions
            
            # Average over non-padded positions
            loss = loss.sum() / mask_flat.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> float:
        """
        Compute character-level accuracy
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            accuracy: float
        """
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
        
        # Compute accuracy
        correct = (predictions == targets).float()
        
        if attention_mask is not None:
            # Only consider non-padded positions
            correct = correct * attention_mask
            accuracy = correct.sum() / attention_mask.sum()
        else:
            accuracy = correct.mean()
        
        return accuracy.item()
    
    def train_step(self, batch: Dict[str, torch.Tensor], 
                   optimizer: torch.optim.Optimizer,
                   clip_grad_norm: float = 1.0) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Dictionary containing input_ids, target_ids, attention_mask
            optimizer: PyTorch optimizer
            clip_grad_norm: Gradient clipping norm
            
        Returns:
            metrics: Dictionary with loss and accuracy
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        logits = self.model(input_ids, attention_mask)
        
        # Compute loss
        loss = self.compute_loss(logits, target_ids, attention_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            accuracy = self.compute_accuracy(logits, target_ids, attention_mask)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single validation step
        
        Args:
            batch: Dictionary containing input_ids, target_ids, attention_mask
            
        Returns:
            metrics: Dictionary with loss and accuracy
        """
        self.model.eval()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        logits = self.model(input_ids, attention_mask)
        
        # Compute loss and accuracy
        loss = self.compute_loss(logits, target_ids, attention_mask)
        accuracy = self.compute_accuracy(logits, target_ids, attention_mask)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def predict(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate predictions
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            predictions: [batch_size, seq_len]
        """
        self.model.eval()
        
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Get model predictions
            logits = self.model(input_ids, attention_mask)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions
    
    def predict_with_constraints(self, input_ids, char_processor, attention_mask=None, temperature=1.0):
        """Make predictions with Vietnamese accent constraints"""
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Use constrained generation
            predictions = self.model.generate_with_constraints(
                input_ids, char_processor, temperature=temperature
            )
            
            return predictions
    
    def save_checkpoint(self, save_path: str, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       epoch: int = 0, metrics: Optional[Dict[str, float]] = None):
        """
        Save model checkpoint
        
        Args:
            save_path: Path to save checkpoint
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (optional)
            epoch: Current epoch
            metrics: Training metrics (optional)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
            }
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str, 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            optimizer: PyTorch optimizer (optional)
            scheduler: Learning rate scheduler (optional)
            
        Returns:
            epoch: Loaded epoch
            metrics: Loaded metrics
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Epoch: {epoch}")
        if metrics:
            print(f"Metrics: {metrics}")
        
        return epoch, metrics 