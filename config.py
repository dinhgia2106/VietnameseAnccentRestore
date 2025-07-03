#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for Vietnamese Accent Restoration System

Centralized configuration management for all components.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
NGRAM_DIR = PROJECT_ROOT / "ngrams"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"

# Model configuration
MODEL_CONFIG: Dict[str, Any] = {
    "max_ngram": 17,
    "atcn": {
        "d_model": 256,
        "num_heads": 8,
        "num_tcn_layers": 6,
        "num_attention_layers": 3,
        "kernel_size": 3,
        "dropout": 0.1
    },
    "context_ranker": {
        "hidden_dim": 128
    }
}

# Training configuration
TRAINING_CONFIG: Dict[str, Any] = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 20,
    "weight_decay": 0.01,
    "gradient_clip_norm": 1.0,
    "validation_split": 0.1,
    "save_every_n_epochs": 5,
    "patience": 3,  # For early stopping
    "lr_scheduler_factor": 0.5
}

# Data preprocessing configuration
PREPROCESSING_CONFIG: Dict[str, Any] = {
    "batch_size": 5000,
    "max_chunk_size": 5 * 1024 * 1024,  # 5MB chunks
    "min_sentence_length": 5,
    "max_sentence_length": 128,
    "valid_char_ratio_threshold": 0.7
}

# Vietnamese character sets
VIETNAMESE_CHARS = {
    "base_chars": set("abcdefghijklmnopqrstuvwxyz"),
    "accented_chars": set(
        "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"
    ),
    "digits": set("0123456789"),
    "punctuation": set(".,!?-:;()[]\"'")
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": PROJECT_ROOT / "logs" / "accent_restore.log"
}

# File paths
FILE_PATHS: Dict[str, Path] = {
    "dictionary": DATA_DIR / "Viet74K_clean.txt",
    "comments_corpus": DATA_DIR / "cleaned_comments.txt",
    "full_corpus": DATA_DIR / "corpus-full.txt",
    "training_dataset": PROCESSED_DATA_DIR / "training_dataset.json",
    "best_model": MODELS_DIR / "best_model.pth",
    "tokenizer_vocab": MODELS_DIR / "tokenizer_vocab.json"
}

# Ensure directories exist
for directory in [DATA_DIR, NGRAM_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGGING_CONFIG["log_file"].parent]:
    directory.mkdir(parents=True, exist_ok=True)

def get_device() -> str:
    """Get the appropriate device for training/inference."""
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def update_config(updates: Dict[str, Any]) -> None:
    """Update configuration with new values."""
    global MODEL_CONFIG, TRAINING_CONFIG, PREPROCESSING_CONFIG
    
    if "model" in updates:
        MODEL_CONFIG.update(updates["model"])
    if "training" in updates:
        TRAINING_CONFIG.update(updates["training"])
    if "preprocessing" in updates:
        PREPROCESSING_CONFIG.update(updates["preprocessing"])

def get_config_summary() -> str:
    """Get a summary of current configuration."""
    return f"""
Vietnamese Accent Restoration System Configuration
==================================================
Project Root: {PROJECT_ROOT}
Data Directory: {DATA_DIR}
N-gram Directory: {NGRAM_DIR}
Models Directory: {MODELS_DIR}

Model Configuration:
- Max N-gram: {MODEL_CONFIG['max_ngram']}
- ATCN d_model: {MODEL_CONFIG['atcn']['d_model']}
- ATCN layers: {MODEL_CONFIG['atcn']['num_tcn_layers']} TCN + {MODEL_CONFIG['atcn']['num_attention_layers']} Attention

Training Configuration:
- Batch size: {TRAINING_CONFIG['batch_size']}
- Learning rate: {TRAINING_CONFIG['learning_rate']}
- Epochs: {TRAINING_CONFIG['epochs']}

Device: {get_device()}
"""

if __name__ == "__main__":
    print(get_config_summary()) 