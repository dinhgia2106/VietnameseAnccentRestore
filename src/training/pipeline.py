"""
A-TCN Training Pipeline
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ..models import ATCN, ATCNTrainer
    from ..data import VietnameseCharProcessor, create_data_loaders
    from .config import TrainingConfig
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models import ATCN, ATCNTrainer
    from data import VietnameseCharProcessor, create_data_loaders
    from training.config import TrainingConfig


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if should stop early"""
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class ATCNTrainingPipeline:
    """
    Complete training pipeline for A-TCN Vietnamese Accent Restoration
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Validate config
        config.validate()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Save config
        self._save_config()
        
        # Initialize components
        self.char_processor = None
        self.model = None
        self.trainer = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.train_loader = None
        self.val_loader = None
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            delta=config.early_stopping_delta
        )
        
        print(f"A-TCN Training Pipeline initialized")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
    
    def _save_config(self):
        """Save training configuration"""
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        print(f"Configuration saved to {config_path}")
    
    def setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        
        # Initialize character processor
        vocab_path = self.config.vocab_path if os.path.exists(self.config.vocab_path) else None
        self.char_processor = VietnameseCharProcessor(vocab_path)
        
        # Create data loaders
        self.train_loader, self.val_loader = create_data_loaders(
            corpus_dir=self.config.corpus_dir,
            char_processor=self.char_processor,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            train_split=self.config.train_split,
            max_files=self.config.max_files,
            max_samples_per_file=self.config.max_samples_per_file,
            num_workers=self.config.num_workers
        )
        
        print(f"Data loaders created:")
        print(f"  Train batches: {len(self.train_loader):,}")
        print(f"  Validation batches: {len(self.val_loader):,}")
        print(f"  Vocabulary size: {self.char_processor.vocab_size}")
    
    def setup_model(self):
        """Setup model, trainer, and optimizer"""
        print("Setting up model...")
        
        # Create model
        self.model = ATCN(
            vocab_size=self.char_processor.vocab_size,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            kernel_size=self.config.kernel_size,
            dropout=self.config.dropout,
            max_dilation=self.config.max_dilation
        )
        
        # Create trainer
        self.trainer = ATCNTrainer(self.model, str(self.device))
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup scheduler
        if self.config.lr_scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_decay_factor,
                patience=self.config.lr_patience,
                verbose=True
            )
        elif self.config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.lr_scheduler == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.num_epochs
            )
        
        print(f"Model setup complete:")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Optimizer: {type(self.optimizer).__name__}")
        print(f"  Scheduler: {type(self.scheduler).__name__}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.config.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            metrics = self.trainer.train_step(
                batch, 
                self.optimizer, 
                self.config.clip_grad_norm
            )
            
            # Update metrics
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            self.global_step += 1
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            avg_accuracy = total_accuracy / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{metrics["loss"]:.4f}',
                'Acc': f'{metrics["accuracy"]:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log periodically
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_training_step(metrics)
        
        # Calculate epoch metrics
        epoch_metrics = {
            'train_loss': total_loss / num_batches,
            'train_accuracy': total_accuracy / num_batches
        }
        
        pbar.close()
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = len(self.val_loader)
        
        # Progress bar
        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False
        )
        
        with torch.no_grad():
            for batch in pbar:
                metrics = self.trainer.val_step(batch)
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                
                # Update progress bar
                avg_loss = total_loss / (pbar.n + 1)
                avg_accuracy = total_accuracy / (pbar.n + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{avg_accuracy:.4f}'
                })
        
        # Calculate validation metrics
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
        
        pbar.close()
        return val_metrics
    
    def _log_training_step(self, metrics: Dict[str, float]):
        """Log training step"""
        print(f"Step {self.global_step}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_data = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict(),
            'char_processor_vocab': {
                'char_to_idx': self.char_processor.char_to_idx,
                'idx_to_char': self.char_processor.idx_to_char,
                'vocab_size': self.char_processor.vocab_size
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint_data, best_path)
            print(f"New best model saved: {best_path}")
        
        # Save vocabulary
        vocab_path = self.output_dir / 'vocab.json'
        self.char_processor.save_vocab(str(vocab_path))
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded: epoch {self.epoch}, step {self.global_step}")
    
    def test_sample_predictions(self, num_samples: int = 5):
        """Test sample predictions"""
        print(f"Testing sample predictions...")
        
        # Get sample batch
        sample_batch = next(iter(self.val_loader))
        input_texts = sample_batch['input_texts'][:num_samples]
        target_texts = sample_batch['target_texts'][:num_samples]
        
        # Generate predictions
        input_ids = sample_batch['input_ids'][:num_samples].to(self.device)
        attention_mask = sample_batch['input_attention_mask'][:num_samples].to(self.device)
        
        predictions = self.trainer.predict(
            input_ids, 
            attention_mask,
            temperature=self.config.temperature,
            top_k=self.config.top_k
        )
        
        # Convert predictions to text
        predicted_texts = []
        for pred in predictions:
            pred_text = self.char_processor.indices_to_text(pred.cpu().tolist())
            predicted_texts.append(pred_text)
        
        # Display results
        print(f"Sample Predictions:")
        for i, (input_text, target_text, pred_text) in enumerate(zip(input_texts, target_texts, predicted_texts)):
            print(f"  Sample {i+1}:")
            print(f"    Input:      '{input_text}'")
            print(f"    Target:     '{target_text}'")
            print(f"    Prediction: '{pred_text}'")
            print()
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.train_history.append(epoch_metrics)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['val_loss'])
            else:
                self.scheduler.step()
            
            # Print epoch summary
            elapsed = time.time() - start_time
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                  f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                  f"Time: {elapsed:.2f}s")
            
            # Check for best model
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch + 1, epoch_metrics, is_best)
            
            # Test sample predictions
            if (epoch + 1) % 10 == 0:
                self.test_sample_predictions()
            
            # Early stopping
            if self.early_stopping(val_metrics['val_loss']):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Final checkpoint
        self.save_checkpoint(self.epoch + 1, epoch_metrics, is_best=True)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.train_history
    
    def run(self):
        """Run complete training pipeline"""
        print("="*60)
        print("A-TCN TRAINING PIPELINE")
        print("="*60)
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        # Train
        history = self.train()
        
        # Save final results
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'config': self.config.to_dict(),
                'history': history,
                'best_val_loss': self.best_val_loss,
                'total_epochs': self.epoch + 1
            }, f, indent=2)
        
        print(f"Training results saved to {results_path}")
        return history 