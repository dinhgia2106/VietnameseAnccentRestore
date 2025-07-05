"""
Vietnamese Character Processor for A-TCN
"""

import json
import string
from typing import List, Dict, Optional
from collections import Counter
import os
import glob
from tqdm import tqdm


class VietnameseCharProcessor:
    """
    Character-level processor cho Vietnamese A-TCN
    Handles character mapping, tokenization, và data preparation
    """
    
    def __init__(self, vocab_path: Optional[str] = None):
        # Vietnamese characters (complete set)
        self.vietnamese_chars = {
            # Basic Latin
            **{c: c for c in string.ascii_lowercase},
            **{c: c for c in string.ascii_uppercase},
            **{c: c for c in string.digits},
            **{c: c for c in string.punctuation},
            ' ': ' ',
            
            # Vietnamese vowels - no accents (input)
            'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u', 'y': 'y',
            
            # Vietnamese vowels - with accents (target)
            'à': 'à', 'á': 'á', 'ả': 'ả', 'ã': 'ã', 'ạ': 'ạ',
            'ă': 'ă', 'ằ': 'ằ', 'ắ': 'ắ', 'ẳ': 'ẳ', 'ẵ': 'ẵ', 'ặ': 'ặ',
            'â': 'â', 'ầ': 'ầ', 'ấ': 'ấ', 'ẩ': 'ẩ', 'ẫ': 'ẫ', 'ậ': 'ậ',
            
            'è': 'è', 'é': 'é', 'ẻ': 'ẻ', 'ẽ': 'ẽ', 'ẹ': 'ẹ',
            'ê': 'ê', 'ề': 'ề', 'ế': 'ế', 'ể': 'ể', 'ễ': 'ễ', 'ệ': 'ệ',
            
            'ì': 'ì', 'í': 'í', 'ỉ': 'ỉ', 'ĩ': 'ĩ', 'ị': 'ị',
            
            'ò': 'ò', 'ó': 'ó', 'ỏ': 'ỏ', 'õ': 'õ', 'ọ': 'ọ',
            'ô': 'ô', 'ồ': 'ồ', 'ố': 'ố', 'ổ': 'ổ', 'ỗ': 'ỗ', 'ộ': 'ộ',
            'ơ': 'ơ', 'ờ': 'ờ', 'ớ': 'ớ', 'ở': 'ở', 'ỡ': 'ỡ', 'ợ': 'ợ',
            
            'ù': 'ù', 'ú': 'ú', 'ủ': 'ủ', 'ũ': 'ũ', 'ụ': 'ụ',
            'ư': 'ư', 'ừ': 'ừ', 'ứ': 'ứ', 'ử': 'ử', 'ữ': 'ữ', 'ự': 'ự',
            
            'ỳ': 'ỳ', 'ý': 'ý', 'ỷ': 'ỷ', 'ỹ': 'ỹ', 'ỵ': 'ỵ',
            
            # Vietnamese consonants
            'd': 'd', 'đ': 'đ',
            
            # Uppercase versions
            'À': 'À', 'Á': 'Á', 'Ả': 'Ả', 'Ã': 'Ã', 'Ạ': 'Ạ',
            'Ă': 'Ă', 'Ằ': 'Ằ', 'Ắ': 'Ắ', 'Ẳ': 'Ẳ', 'Ẵ': 'Ẵ', 'Ặ': 'Ặ',
            'Â': 'Â', 'Ầ': 'Ầ', 'Ấ': 'Ấ', 'Ẩ': 'Ẩ', 'Ẫ': 'Ẫ', 'Ậ': 'Ậ',
            'È': 'È', 'É': 'É', 'Ẻ': 'Ẻ', 'Ẽ': 'Ẽ', 'Ẹ': 'Ẹ',
            'Ê': 'Ê', 'Ề': 'Ề', 'Ế': 'Ế', 'Ể': 'Ể', 'Ễ': 'Ễ', 'Ệ': 'Ệ',
            'Ì': 'Ì', 'Í': 'Í', 'Ỉ': 'Ỉ', 'Ĩ': 'Ĩ', 'Ị': 'Ị',
            'Ò': 'Ò', 'Ó': 'Ó', 'Ỏ': 'Ỏ', 'Õ': 'Õ', 'Ọ': 'Ọ',
            'Ô': 'Ô', 'Ồ': 'Ồ', 'Ố': 'Ố', 'Ổ': 'Ổ', 'Ỗ': 'Ỗ', 'Ộ': 'Ộ',
            'Ơ': 'Ơ', 'Ờ': 'Ờ', 'Ớ': 'Ớ', 'Ở': 'Ở', 'Ỡ': 'Ỡ', 'Ợ': 'Ợ',
            'Ù': 'Ù', 'Ú': 'Ú', 'Ủ': 'Ủ', 'Ũ': 'Ũ', 'Ụ': 'Ụ',
            'Ư': 'Ư', 'Ừ': 'Ừ', 'Ứ': 'Ứ', 'Ử': 'Ử', 'Ữ': 'Ữ', 'Ự': 'Ự',
            'Ỳ': 'Ỳ', 'Ý': 'Ý', 'Ỷ': 'Ỷ', 'Ỹ': 'Ỹ', 'Ỵ': 'Ỵ',
            'Đ': 'Đ'
        }
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        # Build character vocab
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            self._build_vocab()
        
        print(f"Vietnamese Character Processor initialized")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def _build_vocab(self):
        """Build vocabulary from scratch"""
        self.char_to_idx = self.special_tokens.copy()
        self.idx_to_char = {v: k for k, v in self.special_tokens.items()}
        
        # Add characters to vocab
        for i, char in enumerate(sorted(self.vietnamese_chars.keys()), start=len(self.special_tokens)):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char
        
        self.vocab_size = len(self.char_to_idx)
        self.pad_idx = self.special_tokens['<PAD>']
    
    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to character indices"""
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx['<UNK>'])
        return indices
    
    def indices_to_text(self, indices: List[int]) -> str:
        """Convert character indices to text"""
        chars = []
        for idx in indices:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                    chars.append(char)
        return ''.join(chars)
    
    def tokenize_text(self, text: str, max_length: Optional[int] = None) -> Dict:
        """
        Tokenize text with optional padding/truncation
        Returns dict with input_ids, attention_mask
        """
        # Convert to indices
        indices = self.text_to_indices(text)
        
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                # Pad to max_length
                attention_mask = [1] * len(indices) + [0] * (max_length - len(indices))
                indices = indices + [self.pad_idx] * (max_length - len(indices))
        else:
            attention_mask = [1] * len(indices)
        
        return {
            'input_ids': indices,
            'attention_mask': attention_mask,
            'length': len([x for x in attention_mask if x == 1])
        }
    
    def analyze_dataset_characters(self, corpus_dir: str = "corpus_splitted", 
                                 max_files: int = 5) -> Dict:
        """Analyze character distribution in dataset"""
        print("Analyzing character distribution in corpus...")
        
        file_pattern = os.path.join(corpus_dir, "*.json")
        corpus_files = glob.glob(file_pattern)
        
        if not corpus_files:
            print(f"No corpus files found in {corpus_dir}")
            return {}
        
        input_char_counter = Counter()
        target_char_counter = Counter()
        total_samples = 0
        
        # Sample few files for analysis
        sample_files = corpus_files[:max_files]
        
        # Progress bar cho analysis
        analysis_pbar = tqdm(sample_files, desc="Analyzing corpus", unit="file")
        
        for file_path in analysis_pbar:
            analysis_pbar.set_description(f"Analyzing {os.path.basename(file_path)}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Sample items with progress
                sample_items = data[:1000]  # Sample first 1000 items per file
                
                for item in tqdm(sample_items, desc="Counting chars", leave=False):
                    input_text = item["input"]
                    target_text = item["target"]
                    
                    # Count characters
                    for char in input_text:
                        input_char_counter[char] += 1
                    
                    for char in target_text:
                        target_char_counter[char] += 1
                    
                    total_samples += 1
                
                # Update overall progress
                analysis_pbar.set_postfix({'Samples': f'{total_samples:,}'})
                        
            except Exception as e:
                tqdm.write(f"Error processing {file_path}: {e}")
        
        analysis_pbar.close()
        
        # Analysis results
        result = {
            'total_samples_analyzed': total_samples,
            'input_unique_chars': len(input_char_counter),
            'target_unique_chars': len(target_char_counter),
            'input_char_dist': dict(input_char_counter.most_common(20)),
            'target_char_dist': dict(target_char_counter.most_common(20)),
            'unknown_input_chars': [char for char in input_char_counter 
                                   if char not in self.char_to_idx],
            'unknown_target_chars': [char for char in target_char_counter 
                                    if char not in self.char_to_idx]
        }
        
        return result
    
    def save_vocab(self, vocab_path: str = "vietnamese_char_vocab.json"):
        """Save vocabulary to file"""
        vocab_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary saved to {vocab_path}")
    
    def load_vocab(self, vocab_path: str = "vietnamese_char_vocab.json"):
        """Load vocabulary from file"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        self.vocab_size = vocab_data['vocab_size']
        self.special_tokens = vocab_data['special_tokens']
        self.pad_idx = self.special_tokens['<PAD>']
        
        print(f"Vocabulary loaded from {vocab_path}")
        print(f"Vocab size: {self.vocab_size}")
    
    def get_vocab_info(self) -> Dict:
        """Get vocabulary information"""
        return {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'pad_idx': self.pad_idx,
            'sample_chars': list(self.char_to_idx.keys())[:20]
        } 