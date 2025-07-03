#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Vietnamese Accent Restoration System

Common helper functions used across multiple modules.
"""

import os
import re
import json
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper())
    
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True
    )


def normalize_vietnamese_text(text: str) -> str:
    """
    Normalize Vietnamese text for consistent processing.
    
    Args:
        text: Input Vietnamese text
        
    Returns:
        Normalized text
    """
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def remove_vietnamese_accents(text: str) -> str:
    """
    Remove Vietnamese accents using Unicode normalization.
    
    Args:
        text: Vietnamese text with accents
        
    Returns:
        Text without accents
    """
    # Normalize to decomposed form
    text = unicodedata.normalize('NFD', text)
    
    # Remove combining marks (accents)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    return text.lower()


def is_valid_vietnamese_text(text: str, min_ratio: float = 0.7) -> bool:
    """
    Check if text is valid Vietnamese by character ratio.
    
    Args:
        text: Text to validate
        min_ratio: Minimum ratio of valid Vietnamese characters
        
    Returns:
        True if text appears to be Vietnamese
    """
    if not text or len(text.strip()) < 3:
        return False
    
    vietnamese_chars = set(
        'abcdefghijklmnopqrstuvwxyz '
        'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡ'
        'ùúụủũưừứựửữỳýỵỷỹđ0123456789.,!?-:;()[]"\'%'
    )
    
    valid_chars = sum(1 for char in text.lower() if char in vietnamese_chars)
    total_chars = len(text)
    
    return (valid_chars / total_chars) >= min_ratio


def clean_text_for_processing(text: str) -> str:
    """
    Clean text for processing by the accent restoration system.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text ready for processing
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Normalize and strip
    text = normalize_vietnamese_text(text)
    
    return text


def split_into_sentences(text: str, max_length: int = 100) -> List[str]:
    """
    Split text into sentences with maximum length limit.
    
    Args:
        text: Input text
        max_length: Maximum length per sentence
        
    Returns:
        List of sentences
    """
    # Basic sentence splitting
    sentences = re.split(r'[.!?]+', text)
    
    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If sentence is too long, split on commas or spaces
        if len(sentence) > max_length:
            parts = re.split(r'[,;]', sentence)
            for part in parts:
                part = part.strip()
                if len(part) > max_length:
                    # Split on spaces as last resort
                    words = part.split()
                    current = []
                    for word in words:
                        if len(' '.join(current + [word])) > max_length:
                            if current:
                                result.append(' '.join(current))
                                current = [word]
                            else:
                                result.append(word)
                        else:
                            current.append(word)
                    if current:
                        result.append(' '.join(current))
                else:
                    result.append(part)
        else:
            result.append(sentence)
    
    return [s for s in result if len(s.strip()) > 0]


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}


def save_json_file(data: Dict[str, Any], file_path: str) -> bool:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Output file path
        
    Returns:
        True if successful
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved JSON to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def read_text_file(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
    """
    Read text file with multiple encoding attempts.
    
    Args:
        file_path: Path to text file
        encoding: Primary encoding to try
        
    Returns:
        File content or None if failed
    """
    encodings = [encoding, 'utf-8', 'latin1', 'cp1252']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            logger.debug(f"Successfully read {file_path} with encoding {enc}")
            return content
            
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    logger.error(f"Could not read {file_path} with any encoding")
    return None


def batch_process_text_file(file_path: str, 
                           batch_size: int = 5000,
                           encoding: str = 'utf-8'):
    """
    Process large text file in batches.
    
    Args:
        file_path: Path to text file
        batch_size: Number of lines per batch
        encoding: File encoding
        
    Yields:
        Batches of lines
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            batch = []
            for line in f:
                line = line.strip()
                if line:
                    batch.append(line)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            
            if batch:
                yield batch
                
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using character overlap.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    text1 = remove_vietnamese_accents(text1)
    text2 = remove_vietnamese_accents(text2)
    
    if text1 == text2:
        return 1.0
    
    if not text1 or not text2:
        return 0.0
    
    # Character-level similarity
    set1 = set(text1)
    set2 = set(text2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to file
        
    Returns:
        Formatted file size string
    """
    try:
        size = os.path.getsize(file_path)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        
        return f"{size:.1f} TB"
        
    except Exception:
        return "Unknown"


def create_backup(file_path: str) -> bool:
    """
    Create backup of a file.
    
    Args:
        file_path: Path to file to backup
        
    Returns:
        True if backup created successfully
    """
    try:
        if not os.path.exists(file_path):
            return False
        
        backup_path = f"{file_path}.backup"
        
        import shutil
        shutil.copy2(file_path, backup_path)
        
        logger.info(f"Created backup: {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating backup for {file_path}: {e}")
        return False


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys
        
    Returns:
        True if all required keys are present
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        logger.error(f"Missing required config keys: {missing_keys}")
        return False
    
    return True


def progress_bar(current: int, total: int, description: str = "") -> str:
    """
    Create simple text progress bar.
    
    Args:
        current: Current progress
        total: Total items
        description: Progress description
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return f"{description} - No items"
    
    percentage = min(100, int((current / total) * 100))
    bar_length = 30
    filled_length = int(bar_length * current // total)
    
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    
    return f"{description} [{bar}] {percentage}% ({current}/{total})"


if __name__ == "__main__":
    # Test utility functions
    print("Testing Vietnamese text utilities...")
    
    test_text = "Xin chào, tôi là một hệ thống khôi phục dấu tiếng Việt!"
    
    print(f"Original: {test_text}")
    print(f"No accents: {remove_vietnamese_accents(test_text)}")
    print(f"Normalized: {normalize_vietnamese_text(test_text)}")
    print(f"Is valid Vietnamese: {is_valid_vietnamese_text(test_text)}")
    
    sentences = split_into_sentences(test_text, max_length=20)
    print(f"Split sentences: {sentences}")
    
    # Test progress bar
    for i in range(0, 101, 10):
        print(f"\r{progress_bar(i, 100, 'Processing')}", end='')
    print()
    
    print("Utility tests completed!") 