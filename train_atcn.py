#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A-TCN Training Pipeline - Clean Version

Training pipeline cho A-TCN model để xử lý các trường hợp N-gram không handle được
"""

import os
import json
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import glob

from config import MODEL_CONFIG, TRAINING_CONFIG, get_device
from utils import setup_logging, normalize_vietnamese_text, remove_vietnamese_accents, is_valid_vietnamese_text

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


@dataclass
class TrainingStats:
    """Training statistics container."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    learning_rate: float = 0.0


class VietnameseAccentDataset(Dataset):
    """Dataset for Vietnamese accent restoration training."""
    
    def __init__(self, data_pairs: List[Tuple[str, str]], tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = []
        
        logger.info(f"Processing {len(data_pairs)} data pairs...")
        
        valid_pairs = 0
        for no_accent, with_accent in tqdm(data_pairs, desc="Processing data"):
            if self._is_valid_pair(no_accent, with_accent):
                processed = self._process_pair(no_accent, with_accent)
                if processed:
                    self.processed_data.append(processed)
                    valid_pairs += 1
        
        logger.info(f"Dataset ready: {valid_pairs}/{len(data_pairs)} valid pairs")
    
    def _is_valid_pair(self, no_accent: str, with_accent: str) -> bool:
        """Validate a text pair."""
        if not (is_valid_vietnamese_text(no_accent) and is_valid_vietnamese_text(with_accent)):
            return False
        
        target_no_accent = remove_vietnamese_accents(with_accent)
        input_normalized = normalize_vietnamese_text(no_accent)
        
        return target_no_accent.strip() == input_normalized.strip()
    
    def _process_pair(self, no_accent: str, with_accent: str) -> Optional[Dict]:
        """Process a valid text pair into model inputs."""
        try:
            no_accent = normalize_vietnamese_text(no_accent)
            with_accent = normalize_vietnamese_text(with_accent)
            
            input_ids = self.tokenizer.encode(no_accent)
            target_ids = self.tokenizer.encode(with_accent)
            
            if (len(input_ids) > self.max_length or 
                len(target_ids) > self.max_length or
                len(input_ids) < 3 or len(target_ids) < 3):
                return None
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long)
            }
            
        except Exception as e:
            logger.debug(f"Error processing pair: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.processed_data[idx]


def collate_fn(batch: List[Dict], pad_token_id: int = 1) -> Dict:
    """Collate function with dynamic padding."""
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    input_ids_padded = nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    target_ids_padded = nn.utils.rnn.pad_sequence(
        target_ids, batch_first=True, padding_value=pad_token_id
    )
    
    return {
        'input_ids': input_ids_padded,
        'target_ids': target_ids_padded
    }


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor, pad_token_id: int = 1) -> float:
    """Calculate character-level accuracy."""
    mask = (targets != pad_token_id)
    pred_tokens = predictions.argmax(dim=-1)
    correct = (pred_tokens == targets) & mask
    total = mask.sum()
    
    if total == 0:
        return 0.0
    return (correct.sum().float() / total.float()).item()


class ATCNTrainer:
    """Comprehensive trainer for A-TCN model."""
    
    def __init__(self, model, tokenizer, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.to(device)
        self.training_history = []
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def train_epoch(self, dataloader, optimizer, criterion, epoch) -> TrainingStats:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Teacher forcing
            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]
            
            optimizer.zero_grad()
            outputs = self.model(decoder_input)
            
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                decoder_target.reshape(-1)
            )
            
            acc = calculate_accuracy(outputs, decoder_target, self.tokenizer.pad_id)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'acc': f'{total_acc/num_batches:.3f}'
            })
        
        return TrainingStats(
            epoch=epoch,
            train_loss=total_loss / num_batches,
            train_acc=total_acc / num_batches,
            learning_rate=optimizer.param_groups[0]['lr']
        )
    
    def evaluate(self, dataloader, criterion) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
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
                
                acc = calculate_accuracy(outputs, decoder_target, self.tokenizer.pad_id)
                
                total_loss += loss.item()
                total_acc += acc
                num_batches += 1
        
        return total_loss / num_batches, total_acc / num_batches
    
    def train(self, train_dataset, val_dataset=None, epochs=20, 
              batch_size=32, learning_rate=1e-4, save_dir='models'):
        """Main training loop."""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, self.tokenizer.pad_id)
        )
        
        val_loader = None
        if val_dataset:
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        best_val_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader, criterion)
                train_stats.val_loss = val_loss
                train_stats.val_acc = val_acc
                scheduler.step(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.save_model(save_dir, 'best_model.pth')
                    logger.info(f"💾 New best model! Val acc: {val_acc:.3f}, loss: {val_loss:.4f}")
            
            log_msg = f"Epoch {epoch}: Train Loss={train_stats.train_loss:.4f}, Acc={train_stats.train_acc:.3f}"
            if val_loader:
                log_msg += f", Val Loss={val_loss:.4f}, Acc={val_acc:.3f}"
            logger.info(log_msg)
            
            self.training_history.append(train_stats)
            
            if epoch % 5 == 0:
                self.save_model(save_dir, f'checkpoint_epoch_{epoch}.pth')
        
        logger.info("Training completed!")
        return self.training_history
    
    def save_model(self, save_dir: str, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer_vocab_size': self.tokenizer.vocab_size,
            'model_config': MODEL_CONFIG,
            'best_val_loss': self.best_val_loss
        }
        
        filepath = os.path.join(save_dir, filename)
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved: {filepath}")
    
    def predict_text(self, input_text: str) -> str:
        """Generate single prediction from input text."""
        predictions = self.predict_text_multiple(input_text, max_suggestions=1)
        return predictions[0][0] if predictions else input_text
    
    def predict_text_multiple(self, input_text: str, max_suggestions: int = 5, 
                            use_beam_search: bool = True) -> List[Tuple[str, float]]:
        """
        Generate multiple predictions with confidence scores.
        
        Args:
            input_text: Input text without accents
            max_suggestions: Maximum number of suggestions
            use_beam_search: Use beam search for better quality suggestions
            
        Returns:
            List of (prediction, confidence) tuples sorted by confidence
        """
        if not input_text.strip():
            return []
        
        self.model.eval()
        
        try:
            # Tokenize input
            input_tokens = self.tokenizer.encode(input_text)
            input_tensor = torch.tensor([input_tokens], device=self.device)
            
            with torch.no_grad():
                if use_beam_search:
                    return self._beam_search_decode(input_tensor, max_suggestions)
                else:
                    return self._greedy_decode_multiple(input_tensor, max_suggestions)
                    
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return [(input_text, 0.0)]
    
    def _beam_search_decode(self, input_tensor: torch.Tensor, 
                           max_suggestions: int, beam_size: int = None) -> List[Tuple[str, float]]:
        """
        Beam search decoding for multiple high-quality suggestions.
        
        Args:
            input_tensor: Input tensor (batch_size=1, seq_len)
            max_suggestions: Number of suggestions to return
            beam_size: Beam search width (default: max_suggestions * 2)
            
        Returns:
            List of (suggestion, confidence) tuples
        """
        if beam_size is None:
            beam_size = max(max_suggestions * 2, 5)
        
        seq_len = input_tensor.size(1)
        batch_size = 1
        
        # Initialize beam with BOS token
        initial_tokens = input_tensor.clone()
        beams = [(initial_tokens, 0.0, [])]  # (tokens, score, generated_chars)
        
        # Generate predictions
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Collect suggestions from each position
        suggestions = {}
        
        for pos in range(1, seq_len - 1):  # Skip BOS and EOS
            top_probs, top_indices = torch.topk(
                probabilities[0, pos], k=min(beam_size, probabilities.size(-1))
            )
            
            for prob, idx in zip(top_probs, top_indices):
                char_id = idx.item()
                char = self.tokenizer.id_to_char.get(char_id, '<UNK>')
                confidence = prob.item()
                
                if char not in ['<BOS>', '<EOS>', '<PAD>', '<UNK>']:
                    # Build suggestion by replacing character at position
                    suggestion = self._build_suggestion_at_position(
                        input_tensor[0], pos, char_id
                    )
                    
                    if suggestion:
                        # Aggregate confidence for same suggestion
                        if suggestion in suggestions:
                            suggestions[suggestion] = max(suggestions[suggestion], confidence)
                        else:
                            suggestions[suggestion] = confidence
        
        # Sort by confidence and return top suggestions
        sorted_suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)
        return sorted_suggestions[:max_suggestions]
    
    def _greedy_decode_multiple(self, input_tensor: torch.Tensor, 
                               max_suggestions: int) -> List[Tuple[str, float]]:
        """
        Greedy decoding with top-k sampling for multiple suggestions.
        
        Args:
            input_tensor: Input tensor
            max_suggestions: Number of suggestions to return
            
        Returns:
            List of (suggestion, confidence) tuples
        """
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        
        suggestions = {}
        seq_len = input_tensor.size(1)
        
        # Sample from each position
        for pos in range(1, seq_len - 1):
            # Get top k predictions for this position
            k = min(max_suggestions * 2, probabilities.size(-1))
            top_probs, top_indices = torch.topk(probabilities[0, pos], k=k)
            
            for prob, idx in zip(top_probs, top_indices):
                char_id = idx.item()
                char = self.tokenizer.id_to_char.get(char_id, '<UNK>')
                confidence = prob.item()
                
                if char not in ['<BOS>', '<EOS>', '<PAD>', '<UNK>']:
                    suggestion = self._build_suggestion_at_position(
                        input_tensor[0], pos, char_id
                    )
                    
                    if suggestion:
                        if suggestion in suggestions:
                            suggestions[suggestion] = max(suggestions[suggestion], confidence)
                        else:
                            suggestions[suggestion] = confidence
        
        sorted_suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)
        return sorted_suggestions[:max_suggestions]
    
    def _build_suggestion_at_position(self, input_tokens: torch.Tensor, 
                                    pos: int, new_char_id: int) -> str:
        """
        Build a suggestion by replacing character at specific position.
        
        Args:
            input_tokens: Original input tokens
            pos: Position to replace
            new_char_id: New character ID
            
        Returns:
            Generated suggestion string
        """
        try:
            # Create modified tokens
            modified_tokens = input_tokens.clone()
            modified_tokens[pos] = new_char_id
            
            # Decode to text
            suggestion = self.tokenizer.decode(modified_tokens.tolist())
            return suggestion.strip()
            
        except Exception as e:
            logger.debug(f"Error building suggestion: {e}")
            return ""
    
    def predict_word_multiple(self, word: str, max_suggestions: int = 10) -> List[Tuple[str, float]]:
        """
        Generate multiple suggestions for a single word.
        
        Args:
            word: Single word without accents (e.g., "toi")
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of (suggestion, confidence) sorted by confidence
            
        Example:
            predict_word_multiple("toi") -> [("tôi", 0.92), ("tối", 0.78), ("tới", 0.65), ...]
        """
        if not word or not word.strip():
            return []
        
        word = word.strip().lower()
        
        # Use enhanced prediction for single words
        predictions = self.predict_text_multiple(word, max_suggestions, use_beam_search=True)
        
        # Filter to keep only single words and valid Vietnamese
        filtered_predictions = []
        for prediction, confidence in predictions:
            pred = prediction.strip()
            
            # Check if it's a single word
            if ' ' not in pred and len(pred) > 0:
                # Check if it's different from input (has accents)
                if pred != word:
                    filtered_predictions.append((pred, confidence))
        
        return filtered_predictions[:max_suggestions]


def create_training_data_from_corpus(corpus_file: str, output_file: str, max_samples: int = 50000):
    """Create training data from Vietnamese corpus."""
    logger.info(f"Creating training data from: {corpus_file}")
    
    training_pairs = []
    
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                if len(training_pairs) >= max_samples:
                    break
                
                line = line.strip()
                if not line or len(line) < 10:
                    continue
                
                if not is_valid_vietnamese_text(line):
                    continue
                
                with_accent = normalize_vietnamese_text(line)
                no_accent = remove_vietnamese_accents(with_accent)
                
                if with_accent == no_accent:
                    continue
                
                training_pairs.append({
                    "input": no_accent,
                    "target": with_accent
                })
                
                if (line_no + 1) % 10000 == 0:
                    logger.info(f"Processed {line_no + 1} lines, found {len(training_pairs)} pairs")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_pairs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training data saved: {output_file} ({len(training_pairs)} pairs)")
        
    except Exception as e:
        logger.error(f"Error creating training data: {e}")


def load_training_data(file_path: str) -> List[Tuple[str, str]]:
    """Load training data from JSON file."""
    logger.info(f"Loading training data from: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pairs = []
        for item in data:
            if isinstance(item, dict) and 'input' in item and 'target' in item:
                pairs.append((item['input'], item['target']))
        
        logger.info(f"Loaded {len(pairs)} training pairs")
        return pairs
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return []


def load_training_data_from_samples(samples_dir: str, file_pattern: str = "training_data_*.json") -> List[str]:
    """
    Load list of training data files từ samples directory.
    
    Args:
        samples_dir: Directory chứa training data files
        file_pattern: Pattern để match files
        
    Returns:
        List of training data file paths sorted by name
    """
    pattern = os.path.join(samples_dir, file_pattern)
    training_files = glob.glob(pattern)
    training_files.sort()  # Train theo thứ tự
    
    logger.info(f"Found {len(training_files)} training data files in {samples_dir}")
    for i, file_path in enumerate(training_files, 1):
        logger.info(f"  {i:03d}: {os.path.basename(file_path)}")
    
    return training_files


class ProgressiveATCNTrainer(ATCNTrainer):
    """Extended trainer for progressive training từ multiple sample files."""
    
    def __init__(self, model, tokenizer, device: str = 'cpu'):
        super().__init__(model, tokenizer, device)
        
        # Progressive training state
        self.current_file_index = 0
        self.total_files = 0
        self.files_completed = 0
        self.cumulative_training_history = []
        
    def save_progressive_checkpoint(self, save_dir: str, file_index: int, total_files: int):
        """Save checkpoint với progressive training info."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer_vocab_size': self.tokenizer.vocab_size,
            'model_config': MODEL_CONFIG,
            'best_val_loss': self.best_val_loss,
            'current_file_index': file_index,
            'total_files': total_files,
            'files_completed': file_index,
            'cumulative_history': self.cumulative_training_history
        }
        
        checkpoint_file = os.path.join(save_dir, f'progressive_checkpoint_file_{file_index:03d}.pth')
        torch.save(checkpoint, checkpoint_file)
        logger.info(f"Progressive checkpoint saved: {checkpoint_file}")
        
        # Also save as latest
        latest_file = os.path.join(save_dir, 'latest_progressive_checkpoint.pth')
        torch.save(checkpoint, latest_file)
    
    def load_progressive_checkpoint(self, checkpoint_path: str) -> dict:
        """Load progressive training checkpoint."""
        logger.info(f"Loading progressive checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore progressive state
        self.current_file_index = checkpoint.get('current_file_index', 0)
        self.total_files = checkpoint.get('total_files', 0)
        self.files_completed = checkpoint.get('files_completed', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.cumulative_training_history = checkpoint.get('cumulative_history', [])
        
        logger.info(f"Resumed from file {self.files_completed + 1}/{self.total_files}")
        return checkpoint
    
    def train_from_sample_files(self, training_files: List[str], 
                               epochs_per_file: int = 5,
                               batch_size: int = 32,
                               learning_rate: float = 1e-4,
                               save_dir: str = 'models',
                               resume_from: str = None):
        """
        Train progressively từ multiple sample files.
        
        Args:
            training_files: List of training data JSON files
            epochs_per_file: Epochs to train on each file
            batch_size: Batch size
            learning_rate: Learning rate
            save_dir: Directory to save models
            resume_from: Checkpoint file to resume from
        """
        self.total_files = len(training_files)
        
        # Resume from checkpoint if provided
        start_file_index = 0
        if resume_from and os.path.exists(resume_from):
            self.load_progressive_checkpoint(resume_from)
            start_file_index = self.files_completed
            logger.info(f"Resuming training from file {start_file_index + 1}/{self.total_files}")
        
        # Setup optimizer (recreate nếu resuming)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Starting progressive training on {self.total_files} files")
        logger.info(f"Epochs per file: {epochs_per_file}")
        
        for file_index in range(start_file_index, self.total_files):
            training_file = training_files[file_index]
            self.current_file_index = file_index
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Training on file {file_index + 1}/{self.total_files}: {os.path.basename(training_file)}")
            logger.info(f"{'='*80}")
            
            # Load data for current file
            training_pairs = load_training_data(training_file)
            if not training_pairs:
                logger.warning(f"No training data in {training_file}, skipping")
                continue
            
            # Create dataset
            dataset = VietnameseAccentDataset(training_pairs, self.tokenizer)
            if len(dataset) == 0:
                logger.warning(f"No valid training data after processing {training_file}, skipping")
                continue
            
            # Split train/validation
            val_size = max(1, len(dataset) // 10)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            logger.info(f"File {file_index + 1}: Train={len(train_dataset)}, Val={len(val_dataset)}")
            
            # Train on current file
            file_history = self.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=epochs_per_file,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_dir=save_dir
            )
            
            # Track cumulative progress
            self.cumulative_training_history.extend(file_history)
            self.files_completed = file_index + 1
            
            # Save progressive checkpoint
            self.save_progressive_checkpoint(save_dir, file_index + 1, self.total_files)
            
            # Log overall progress
            logger.info(f"Completed file {file_index + 1}/{self.total_files}")
            logger.info(f"Current best val loss: {self.best_val_loss:.4f}")
            
            # Cleanup to free memory
            del dataset, train_dataset, val_dataset, training_pairs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"\n{'='*80}")
        logger.info("Progressive training completed!")
        logger.info(f"Trained on {self.files_completed}/{self.total_files} files")
        logger.info(f"Total epochs: {self.files_completed * epochs_per_file}")
        logger.info(f"Final best val loss: {self.best_val_loss:.4f}")
        logger.info(f"{'='*80}")
        
        return self.cumulative_training_history


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train A-TCN Vietnamese Accent Restoration')
    
    # Corpus splitting arguments
    parser.add_argument('--split_corpus', action='store_true', help='Split corpus into sample files first')
    parser.add_argument('--corpus_files', nargs='+', 
                       default=['data/cleaned_comments.txt', 'data/corpus-full.txt'],
                       help='Corpus files to split')
    parser.add_argument('--samples_per_file', type=int, default=500000,
                       help='Samples per file when splitting')
    
    # Data arguments  
    parser.add_argument('--data_file', type=str, help='Single training data JSON file')
    parser.add_argument('--samples_dir', type=str, default='processed_data/samples/training_data',
                       help='Directory containing multiple training data files')
    parser.add_argument('--create_data', action='store_true', help='Create training data from corpus')
    parser.add_argument('--max_samples', type=int, default=50000)
    
    # Training mode
    parser.add_argument('--progressive_training', action='store_true', 
                       help='Train progressively from multiple sample files')
    parser.add_argument('--resume_from', type=str, help='Checkpoint file to resume progressive training')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--epochs_per_file', type=int, default=5, 
                       help='Epochs per file in progressive training')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--test_text', type=str, help='Test text after training')
    
    args = parser.parse_args()
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Step 1: Split corpus if requested
    if args.split_corpus:
        logger.info("Splitting corpus into sample files...")
        from corpus_splitter import CorpusSplitter
        
        splitter = CorpusSplitter(samples_per_file=args.samples_per_file)
        sample_files = splitter.split_corpus(args.corpus_files)
        
        if sample_files:
            logger.info("Creating training data from sample files...")
            training_dir = os.path.join(os.path.dirname(sample_files[0]), "training_data")
            os.makedirs(training_dir, exist_ok=True)
            
            for i, sample_file in enumerate(sample_files, 1):
                training_file = os.path.join(training_dir, f"training_data_{i:03d}.json")
                splitter.create_training_pairs_from_file(sample_file, training_file)
            
            logger.info(f"Training data files created in: {training_dir}")
        return
    
    # Step 2: Create training data if requested
    if args.create_data:
        if not args.corpus_files[0] or not os.path.exists(args.corpus_files[0]):
            logger.error("Need valid corpus file to create training data")
            return
        
        output_file = args.data_file or 'processed_data/training_dataset.json'
        create_training_data_from_corpus(args.corpus_files[0], output_file, args.max_samples)
        return
    
    # Step 3: Training
    logger.info("Creating model...")
    from atcn_model import create_model
    model, tokenizer, _ = create_model(device=device)
    
    if args.progressive_training:
        # Progressive training from multiple files
        logger.info("Using progressive training mode...")
        
        # Load training files
        training_files = load_training_data_from_samples(args.samples_dir)
        if not training_files:
            logger.error(f"No training files found in {args.samples_dir}")
            return
        
        # Create progressive trainer
        trainer = ProgressiveATCNTrainer(model, tokenizer, device)
        
        # Train progressively
        trainer.train_from_sample_files(
            training_files=training_files,
            epochs_per_file=args.epochs_per_file,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_dir=args.model_dir,
            resume_from=args.resume_from
        )
    
    else:
        # Single file training (original mode)
        if not args.data_file:
            logger.error("Need --data_file for single file training")
            return
        
        if not os.path.exists(args.data_file):
            logger.error(f"Training data not found: {args.data_file}")
            return
        
        training_pairs = load_training_data(args.data_file)
        if not training_pairs:
            logger.error("No training data loaded")
            return
        
        # Create datasets
        full_dataset = VietnameseAccentDataset(training_pairs, tokenizer)
        
        # Split train/validation
        val_size = int(len(full_dataset) * 0.1)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Train model
        trainer = ATCNTrainer(model, tokenizer, device)
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_dir=args.model_dir
        )
    
    # Test if requested
    if args.test_text:
        # Use appropriate trainer type
        if args.progressive_training:
            trainer_for_test = ProgressiveATCNTrainer(model, tokenizer, device)
        else:
            trainer_for_test = ATCNTrainer(model, tokenizer, device)
            
        predicted = trainer_for_test.predict_text(args.test_text)
        logger.info(f"Input: {args.test_text}")
        logger.info(f"Prediction: {predicted}")


if __name__ == "__main__":
    main() 