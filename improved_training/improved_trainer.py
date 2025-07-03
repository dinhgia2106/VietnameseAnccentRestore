#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved A-TCN Trainer với proper validation và metrics.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple
import logging
from tqdm import tqdm

from train_atcn_clean import ATCNTrainer, VietnameseAccentDataset, load_training_data, collate_fn
from atcn_model_clean import create_model
from config import get_device

logger = logging.getLogger(__name__)

class ImprovedATCNTrainer(ATCNTrainer):
    """Enhanced trainer với better metrics và validation."""
    
    def __init__(self, model, tokenizer, device: str = 'cpu'):
        super().__init__(model, tokenizer, device)
        self.val_history = []
        
    def evaluate_with_metrics(self, dataloader, criterion):
        """Enhanced evaluation với multiple metrics."""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                decoder_input = target_ids[:, :-1]
                decoder_target = target_ids[:, 1:]
                
                outputs = self.model(decoder_input)
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    decoder_target.reshape(-1)
                )
                
                total_loss += loss.item()
                
                # Get predictions for metrics
                pred_ids = torch.argmax(outputs, dim=-1)
                
                # Decode for metrics calculation
                for i in range(pred_ids.size(0)):
                    pred_text = self.tokenizer.decode(pred_ids[i].cpu().tolist())
                    target_text = self.tokenizer.decode(decoder_target[i].cpu().tolist())
                    
                    predictions.append(pred_text.strip())
                    targets.append(target_text.strip())
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate additional metrics
        char_accuracy = self.calculate_character_accuracy(predictions, targets)
        word_accuracy = self.calculate_word_accuracy(predictions, targets)
        
        return {
            'loss': avg_loss,
            'char_accuracy': char_accuracy,
            'word_accuracy': word_accuracy,
            'num_samples': len(predictions)
        }
    
    def calculate_character_accuracy(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate character-level accuracy."""
        if len(predictions) != len(targets):
            return 0.0
        
        total_chars = 0
        correct_chars = 0
        
        for pred, target in zip(predictions, targets):
            total_chars += max(len(pred), len(target))
            
            for i in range(min(len(pred), len(target))):
                if pred[i] == target[i]:
                    correct_chars += 1
        
        return correct_chars / total_chars if total_chars > 0 else 0.0
    
    def calculate_word_accuracy(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate word-level exact match accuracy."""
        if len(predictions) != len(targets):
            return 0.0
        
        correct = sum(1 for pred, target in zip(predictions, targets) 
                      if pred.strip() == target.strip())
        
        return correct / len(predictions)
    
    def train_improved(self, train_data_file: str, val_data_file: str, 
                      epochs: int = 20, batch_size: int = 32, 
                      learning_rate: float = 1e-4, save_dir: str = 'models'):
        """Improved training với proper validation."""
        
        # Load datasets
        train_pairs = load_training_data(train_data_file)
        val_pairs = load_training_data(val_data_file)
        
        train_dataset = VietnameseAccentDataset(train_pairs, self.tokenizer)
        val_dataset = VietnameseAccentDataset(val_pairs, self.tokenizer)
        
        print(f"Train dataset: {len(train_dataset):,} samples")
        print(f"Val dataset: {len(val_dataset):,} samples")
        print(f"Val ratio: {len(val_dataset)/(len(train_dataset)+len(val_dataset))*100:.1f}%")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_id)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_id)
        )
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5, verbose=True
        )
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 5
        
        print(f"Starting improved training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_stats = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validate with metrics
            val_metrics = self.evaluate_with_metrics(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_metrics['loss'])
            
            # Log detailed metrics
            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {train_stats.train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Char Acc: {val_metrics['char_accuracy']:.3f}")
            print(f"  Val Word Acc: {val_metrics['word_accuracy']:.3f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model based on character accuracy
            if val_metrics['char_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['char_accuracy']
                self.save_model(save_dir, 'best_model.pth')
                print(f"  ✓ New best model! Char Acc: {best_val_acc:.3f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping after {max_patience} epochs without improvement")
                break
            
            # Track history
            self.val_history.append({
                'epoch': epoch,
                'train_loss': train_stats.train_loss,
                'val_loss': val_metrics['loss'],
                'val_char_acc': val_metrics['char_accuracy'],
                'val_word_acc': val_metrics['word_accuracy']
            })
            
            print()
        
        print("Improved training completed!")
        print(f"Best validation character accuracy: {best_val_acc:.3f}")
        
        return self.val_history

def main():
    """Run improved training."""
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    model, tokenizer, _ = create_model(device=device)
    
    # Create improved trainer
    trainer = ImprovedATCNTrainer(model, tokenizer, device)
    
    # Train with improved data splits
    train_file = "processed_data/improved_splits/train_data.json"
    val_file = "processed_data/improved_splits/val_data.json"
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print("❌ Improved data splits not found. Run create_proper_train_val_split() first.")
        return
    
    # Train
    history = trainer.train_improved(
        train_data_file=train_file,
        val_data_file=val_file,
        epochs=15,
        batch_size=32,
        learning_rate=5e-5,  # Lower LR
        save_dir='models/improved'
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
