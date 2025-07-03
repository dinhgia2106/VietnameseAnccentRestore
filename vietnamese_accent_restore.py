#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnamese Accent Restoration System using N-gram approach

This module provides intelligent accent restoration for Vietnamese text using
pre-computed n-gram statistics from a large Vietnamese dictionary.

Author: Vietnamese Accent Restore Team
Version: 2.0
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietnameseAccentRestore:
    """
    Vietnamese Accent Restoration System using N-gram approach.
    
    This class provides intelligent suggestion of Vietnamese accents based on
    n-gram statistics computed from a large Vietnamese dictionary corpus.
    
    Attributes:
        max_ngram (int): Maximum n-gram length to use
        ngram_data (Dict): Loaded n-gram statistics
    """
    
    def __init__(self, max_ngram: int = 17, ngram_dir: str = "ngrams"):
        """
        Initialize the accent restoration system.
        
        Args:
            max_ngram: Maximum n-gram length to consider
            ngram_dir: Directory containing n-gram JSON files
        """
        self.max_ngram = max_ngram
        self.ngram_dir = ngram_dir
        self.ngram_data: Dict[int, Dict[str, List[str]]] = {}
        
        self._load_ngrams()
    
    def _load_ngrams(self) -> None:
        """Load n-gram data from JSON files."""
        logger.info("Loading n-gram data...")
        
        for n in range(1, self.max_ngram + 1):
            ngram_file = os.path.join(self.ngram_dir, f"{n}_gram.json")
            
            if not os.path.exists(ngram_file):
                logger.warning(f"N-gram file not found: {ngram_file}")
                continue
                
            try:
                with open(ngram_file, 'r', encoding='utf-8') as f:
                    self.ngram_data[n] = json.load(f)
                logger.info(f"Loaded {len(self.ngram_data[n])} {n}-grams")
                
            except Exception as e:
                logger.error(f"Error loading {ngram_file}: {e}")
        
        logger.info("N-gram data loading completed!")
    
    @staticmethod
    def remove_accents(text: str) -> str:
        """
        Remove Vietnamese accents from text.
        
        Args:
            text: Input text with accents
            
        Returns:
            Text without accents in lowercase
        """
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        return text.lower()
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        # Remove special characters, keep only letters, numbers, spaces, basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip().lower()
    
    def find_suggestions(self, input_text: str, max_suggestions: int = 10) -> List[Tuple[str, float]]:
        """
        Find accent suggestions for input text.
        
        This method uses a sophisticated strategy:
        1. Find exact matches for the full input
        2. Find the input as a subsequence in longer n-grams
        3. Use partial matching only when needed (avoiding random 1-gram combinations)
        
        Args:
            input_text: Text without accents
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of (suggestion, confidence_score) tuples sorted by confidence
        """
        clean_input = self.clean_text(input_text)
        input_words = clean_input.split()
        
        if not input_words:
            return []
        
        no_accent_input = self.remove_accents(clean_input)
        suggestions: Dict[str, float] = {}
        
        # Strategy 1: Find exact matches
        exact_suggestions = self._find_exact_matches(no_accent_input)
        for suggestion in exact_suggestions:
            confidence = self._calculate_confidence(suggestion, clean_input, len(input_words))
            suggestions[suggestion] = max(suggestions.get(suggestion, 0), confidence)
        
        # Strategy 2: Find as subsequence in longer n-grams
        subseq_suggestions = self._find_subsequence_matches(input_words)
        for suggestion in subseq_suggestions:
            confidence = self._calculate_confidence(suggestion, clean_input, len(input_words))
            suggestions[suggestion] = max(suggestions.get(suggestion, 0), confidence)
        
        # Strategy 3: Partial matches (only if we have few suggestions)
        if len(suggestions) < 3:
            partial_suggestions = self._find_partial_matches(input_words)
            for suggestion in partial_suggestions:
                confidence = self._calculate_confidence(suggestion, clean_input, len(input_words))
                suggestions[suggestion] = max(suggestions.get(suggestion, 0), confidence)
        
        # Sort by confidence and return top suggestions
        sorted_suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)
        return sorted_suggestions[:max_suggestions]
    
    def _find_exact_matches(self, no_accent_input: str) -> List[str]:
        """Find exact matches for the full input."""
        input_words = no_accent_input.split()
        n = len(input_words)
        
        if n in self.ngram_data and no_accent_input in self.ngram_data[n]:
            return self.ngram_data[n][no_accent_input]
        
        return []
    
    def _find_subsequence_matches(self, input_words: List[str]) -> List[str]:
        """
        Find input as a subsequence in longer n-grams.
        
        Example: "may bay" found in "may bay hac lanh" -> extract "máy bay"
        """
        suggestions = []
        input_len = len(input_words)
        
        # Search in longer n-grams
        for n in range(input_len + 1, self.max_ngram + 1):
            if n not in self.ngram_data:
                continue
            
            for ngram_key, ngram_values in self.ngram_data[n].items():
                ngram_words = ngram_key.split()
                
                # Check if input is a contiguous subsequence
                if self._is_contiguous_subsequence(input_words, ngram_words):
                    # Extract corresponding part from each ngram value
                    for value in ngram_values:
                        extracted = self._extract_corresponding_subsequence(
                            input_words, ngram_words, value
                        )
                        if extracted and extracted not in suggestions:
                            suggestions.append(extracted)
        
        return suggestions
    
    def _find_partial_matches(self, input_words: List[str]) -> List[str]:
        """
        Find partial matches using n-grams >= 2 to avoid meaningless 1-gram combinations.
        """
        suggestions = []
        
        # Only use n-grams of length 2 or more to avoid random combinations
        for n in range(2, len(input_words) + 1):
            if n not in self.ngram_data:
                continue
            
            for i in range(len(input_words) - n + 1):
                partial_input = ' '.join(input_words[i:i+n])
                partial_no_accent = self.remove_accents(partial_input)
                
                if partial_no_accent in self.ngram_data[n]:
                    for suggestion_part in self.ngram_data[n][partial_no_accent]:
                        full_suggestion = self._replace_partial_in_text(
                            input_words, i, i+n, suggestion_part
                        )
                        if full_suggestion and full_suggestion not in suggestions:
                            suggestions.append(full_suggestion)
        
        return suggestions
    
    @staticmethod
    def _is_contiguous_subsequence(subsequence: List[str], sequence: List[str]) -> bool:
        """Check if subsequence is a contiguous part of sequence."""
        subseq_len = len(subsequence)
        seq_len = len(sequence)
        
        if subseq_len > seq_len:
            return False
        
        for i in range(seq_len - subseq_len + 1):
            if sequence[i:i+subseq_len] == subsequence:
                return True
        
        return False
    
    @staticmethod
    def _extract_corresponding_subsequence(input_words: List[str], 
                                         ngram_words: List[str], 
                                         ngram_value: str) -> Optional[str]:
        """Extract the part of ngram_value corresponding to input_words."""
        input_len = len(input_words)
        ngram_len = len(ngram_words)
        ngram_value_words = ngram_value.split()
        
        if len(ngram_value_words) != ngram_len:
            return None
        
        # Find position of input_words in ngram_words
        for i in range(ngram_len - input_len + 1):
            if ngram_words[i:i+input_len] == input_words:
                extracted_words = ngram_value_words[i:i+input_len]
                return ' '.join(extracted_words)
        
        return None
    
    @staticmethod
    def _replace_partial_in_text(input_words: List[str], 
                               start_idx: int, 
                               end_idx: int, 
                               replacement: str) -> str:
        """Replace a partial sequence in input_words with replacement."""
        result_words = input_words.copy()
        replacement_words = replacement.split()
        result_words[start_idx:end_idx] = replacement_words
        return ' '.join(result_words)
    
    def _calculate_confidence(self, suggestion: str, original: str, input_length: int) -> float:
        """
        Calculate confidence score for a suggestion.
        
        Args:
            suggestion: The suggested text with accents
            original: Original input text
            input_length: Number of words in input
            
        Returns:
            Confidence score (higher is better)
        """
        suggestion_no_accent = self.remove_accents(suggestion)
        original_no_accent = self.remove_accents(original)
        
        # Base score for exact match
        if suggestion_no_accent == original_no_accent:
            base_score = 10.0
        else:
            # Heavy penalty for non-exact matches
            return 0.1
        
        # Bonus for longer n-grams (prefer longer context)
        suggestion_length = len(suggestion.split())
        length_bonus = suggestion_length * 2.0
        
        # Extra bonus for exact length match
        if suggestion_length == input_length:
            length_bonus += 5.0
        
        # Add frequency-based bonus
        frequency_bonus = self._get_frequency_bonus(suggestion)
        
        return base_score + length_bonus + frequency_bonus
    
    def _get_frequency_bonus(self, suggestion: str) -> float:
        """
        Calculate frequency bonus based on word commonality.
        
        Args:
            suggestion: The suggested text
            
        Returns:
            Frequency bonus score
        """
        # Common Vietnamese words get higher bonus
        common_words = {
            # Pronouns - very common
            'tôi': 10.0, 'bạn': 8.0, 'chúng tôi': 7.0, 'họ': 6.0,
            
            # Common actions
            'đi': 9.0, 'có': 9.0, 'là': 9.0, 'làm': 8.0, 'học': 8.0,
            'ăn': 7.0, 'uống': 6.0, 'ngủ': 6.0, 'chơi': 6.0,
            
            # Time words
            'sáng': 7.0, 'tối': 6.0, 'ngày': 8.0, 'đêm': 6.0, 'tuần': 6.0,
            
            # Transport & objects
            'máy bay': 9.0, 'xe': 8.0, 'nhà': 8.0, 'trường': 7.0,
            'sách': 6.0, 'bàn': 6.0, 'ghế': 5.0,
            
            # Weather & nature  
            'mây bay': 3.0,  # Less common than máy bay
            'mây': 5.0, 'nước': 7.0, 'cây': 6.0,
            
            # Colors
            'màu đỏ': 8.0, 'màu xanh': 7.0, 'màu vàng': 6.0,
            'màu': 7.0, 'đỏ': 6.0, 'xanh': 6.0,
            
            # Body parts  
            'máu': 4.0,  # Less common than màu
            'tay': 7.0, 'chân': 6.0, 'đầu': 7.0, 'mắt': 6.0,
            
            # Greetings & politeness
            'cảm ơn': 9.0, 'xin chào': 8.0, 'chào': 7.0,
            'xin lỗi': 7.0, 'cám ơn': 3.0,  # Less common spelling
            
            # Education
            'học sinh': 8.0, 'giáo viên': 7.0, 'trường học': 7.0,
            'lớp': 6.0, 'bài': 6.0,
            
            # Directions
            'tới': 6.0, 'đến': 8.0, 'về': 7.0, 'ra': 6.0, 'vào': 6.0,
            'tỏi': 3.0,  # garlic - less common
            'tội': 4.0,  # sin/crime - less common than tôi/tới
            
            # Common mistakes to deprioritize
            'tòi': 2.0, 'tói': 1.0, 'tồi': 3.0,  # Less common variations
            'mấy': 6.0, 'mây': 5.0, 'may': 4.0,  # "may" without accent less common
            'cắm': 5.0, 'cảm': 7.0, 'cam': 4.0,  # "cam" fruit vs "cảm" feeling
        }
        
        # Check for exact matches first
        suggestion_lower = suggestion.lower().strip()
        if suggestion_lower in common_words:
            return common_words[suggestion_lower]
        
        # Check individual words for multi-word suggestions
        words = suggestion_lower.split()
        if len(words) > 1:
            total_bonus = 0.0
            word_count = 0
            
            for word in words:
                if word in common_words:
                    total_bonus += common_words[word]
                    word_count += 1
            
            # Average bonus for multi-word phrases
            if word_count > 0:
                return total_bonus / len(words)
        
        # Default bonus for any valid suggestion
        return 1.0
    
    def suggest_single_word(self, word: str, max_suggestions: int = 5) -> List[str]:
        """
        Convenience method to get suggestions for a single word.
        
        Args:
            word: Single word without accents
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested words with accents
        """
        suggestions = self.find_suggestions(word, max_suggestions)
        return [suggestion for suggestion, _ in suggestions]
    
    def interactive_demo(self) -> None:
        """Run interactive demonstration of the system."""
        print("VIETNAMESE ACCENT RESTORATION SYSTEM V2")
        print("=" * 80)
        print("Nhập từ/cụm từ không dấu để nhận gợi ý (gõ 'quit' để thoát)")
        print()
        
        while True:
            try:
                user_input = input("Nhập từ cần gợi ý: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Tạm biệt!")
                    break
                
                if not user_input:
                    continue
                
                suggestions = self.find_suggestions(user_input, max_suggestions=10)
                
                if suggestions:
                    print(f"\nGợi ý cho '{user_input}':")
                    for i, (suggestion, confidence) in enumerate(suggestions, 1):
                        print(f"  {i}. {suggestion} (độ tin cậy: {confidence:.1f})")
                else:
                    print(f"Không tìm thấy gợi ý cho '{user_input}'")
                
                print()
                
            except KeyboardInterrupt:
                print("\nTạm biệt!")
                break
            except Exception as e:
                logger.error(f"Error in interactive demo: {e}")


def test_system_performance() -> None:
    """Test the system with common Vietnamese phrases."""
    restore = VietnameseAccentRestore(max_ngram=5)
    
    test_cases = [
        "may bay",
        "cam on", 
        "an com",
        "hoc sinh",
        "tieu hoc",
        "truong hoc",
        "nha truong",
        "giao vien",
        "co giao"
    ]
    
    print("PERFORMANCE TEST")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\nInput: '{test_case}'")
        suggestions = restore.find_suggestions(test_case, max_suggestions=5)
        
        if suggestions:
            for i, (suggestion, confidence) in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion} (tin cậy: {confidence:.1f})")
        else:
            print("  Không có gợi ý")


def main() -> None:
    """Main function to run the system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vietnamese Accent Restoration System')
    parser.add_argument('--test', action='store_true', help='Run performance test')
    parser.add_argument('--max_ngram', type=int, default=5, help='Maximum n-gram length')
    parser.add_argument('--ngram_dir', type=str, default='ngrams', help='N-gram directory')
    
    args = parser.parse_args()
    
    # Initialize system
    restore = VietnameseAccentRestore(max_ngram=args.max_ngram, ngram_dir=args.ngram_dir)
    
    if args.test:
        test_system_performance()
    else:
        restore.interactive_demo()


if __name__ == "__main__":
    main() 