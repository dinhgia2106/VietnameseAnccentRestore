"""
Vietnamese Character Dataset for A-TCN Training
"""

import json
import os
import glob
import random
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
try:
    from .processor import VietnameseCharProcessor
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from processor import VietnameseCharProcessor


class VietnameseCharDataset(Dataset):
    """
    PyTorch Dataset for Vietnamese character-level accent restoration
    """
    
    def __init__(self, corpus_dir: str, char_processor: VietnameseCharProcessor, 
                 max_length: int = 512, max_files: Optional[int] = None,
                 max_samples_per_file: int = 10000):
        """
        Initialize dataset
        
        Args:
            corpus_dir: Directory containing JSON corpus files
            char_processor: Vietnamese character processor
            max_length: Maximum sequence length
            max_files: Maximum number of files to load (None for all)
            max_samples_per_file: Maximum samples per file
        """
        self.corpus_dir = corpus_dir
        self.char_processor = char_processor
        self.max_length = max_length
        self.data = []
        
        print(f"Loading data from {corpus_dir}...")
        self._load_data(max_files, max_samples_per_file)
        
        print(f"Dataset initialized with {len(self.data)} samples")
    
    def _load_data(self, max_files: Optional[int], max_samples_per_file: int):
        """Load data from corpus files"""
        # Get list of corpus files
        file_pattern = os.path.join(self.corpus_dir, "*.json")
        corpus_files = sorted(glob.glob(file_pattern))
        
        if not corpus_files:
            raise FileNotFoundError(f"No corpus files found in {self.corpus_dir}")
        
        # Limit number of files if specified
        if max_files:
            corpus_files = corpus_files[:max_files]
        
        print(f"Found {len(corpus_files)} corpus files")
        
        # Load data from each file
        for file_path in tqdm(corpus_files, desc="Loading files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                # Limit samples per file
                if len(file_data) > max_samples_per_file:
                    # Random sample to maintain diversity
                    file_data = random.sample(file_data, max_samples_per_file)
                
                # Process each sample
                for item in file_data:
                    if "input" in item and "target" in item:
                        input_text = item["input"]
                        target_text = item["target"]
                        
                        # Skip if too long
                        if len(input_text) > self.max_length or len(target_text) > self.max_length:
                            continue
                        
                        self.data.append({
                            'input': input_text,
                            'target': target_text
                        })
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx]
        input_text = sample['input']
        target_text = sample['target']
        
        # Tokenize input and target
        input_tokens = self.char_processor.tokenize_text(input_text, self.max_length)
        target_tokens = self.char_processor.tokenize_text(target_text, self.max_length)
        
        return {
            'input_ids': torch.tensor(input_tokens['input_ids'], dtype=torch.long),
            'target_ids': torch.tensor(target_tokens['input_ids'], dtype=torch.long),
            'input_attention_mask': torch.tensor(input_tokens['attention_mask'], dtype=torch.long),
            'target_attention_mask': torch.tensor(target_tokens['attention_mask'], dtype=torch.long),
            'input_text': input_text,
            'target_text': target_text
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    # Extract all fields
    input_ids = torch.stack([item['input_ids'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    input_attention_mask = torch.stack([item['input_attention_mask'] for item in batch])
    target_attention_mask = torch.stack([item['target_attention_mask'] for item in batch])
    
    # Text data (for debugging/visualization)
    input_texts = [item['input_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'input_attention_mask': input_attention_mask,
        'target_attention_mask': target_attention_mask,
        'input_texts': input_texts,
        'target_texts': target_texts
    }


def create_data_loaders(corpus_dir: str, 
                       char_processor: VietnameseCharProcessor,
                       batch_size: int = 32,
                       max_length: int = 512,
                       train_split: float = 0.9,
                       max_files: Optional[int] = None,
                       max_samples_per_file: int = 10000,
                       num_workers: int = 0,
                       shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        corpus_dir: Directory containing corpus files
        char_processor: Vietnamese character processor
        batch_size: Batch size for training
        max_length: Maximum sequence length
        train_split: Ratio of training data (0.9 = 90% train, 10% val)
        max_files: Maximum number of files to load
        max_samples_per_file: Maximum samples per file
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    print("Creating data loaders...")
    
    # Create full dataset
    full_dataset = VietnameseCharDataset(
        corpus_dir=corpus_dir,
        char_processor=char_processor,
        max_length=max_length,
        max_files=max_files,
        max_samples_per_file=max_samples_per_file
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    # Create random split
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def analyze_dataset_stats(data_loader: DataLoader) -> Dict:
    """
    Analyze dataset statistics
    
    Args:
        data_loader: DataLoader to analyze
    
    Returns:
        statistics: Dictionary with dataset statistics
    """
    print("Analyzing dataset statistics...")
    
    total_samples = 0
    total_input_chars = 0
    total_target_chars = 0
    input_lengths = []
    target_lengths = []
    
    for batch in tqdm(data_loader, desc="Analyzing batches"):
        batch_size = batch['input_ids'].size(0)
        total_samples += batch_size
        
        # Count non-padded characters
        input_mask = batch['input_attention_mask']
        target_mask = batch['target_attention_mask']
        
        input_lengths.extend(input_mask.sum(dim=1).tolist())
        target_lengths.extend(target_mask.sum(dim=1).tolist())
        
        total_input_chars += input_mask.sum().item()
        total_target_chars += target_mask.sum().item()
    
    stats = {
        'total_samples': total_samples,
        'total_input_chars': total_input_chars,
        'total_target_chars': total_target_chars,
        'avg_input_length': sum(input_lengths) / len(input_lengths),
        'avg_target_length': sum(target_lengths) / len(target_lengths),
        'max_input_length': max(input_lengths),
        'max_target_length': max(target_lengths),
        'min_input_length': min(input_lengths),
        'min_target_length': min(target_lengths)
    }
    
    print(f"Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Avg input length: {stats['avg_input_length']:.2f}")
    print(f"  Avg target length: {stats['avg_target_length']:.2f}")
    print(f"  Max input length: {stats['max_input_length']}")
    print(f"  Max target length: {stats['max_target_length']}")
    
    return stats 