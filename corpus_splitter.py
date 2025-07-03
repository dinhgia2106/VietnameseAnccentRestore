#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpus Splitter - Chia corpus lớn thành sample files để training

Chia cleaned_comments.txt và corpus-full.txt thành các file samples
không trùng nhau, mỗi file ~500k samples.
"""

import os
import random
import logging
from typing import List, Iterator
from pathlib import Path

from utils import setup_logging, is_valid_vietnamese_text, normalize_vietnamese_text

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


class CorpusSplitter:
    """Chia corpus lớn thành sample files để training."""
    
    def __init__(self, samples_per_file: int = 500000, min_line_length: int = 10):
        """
        Initialize corpus splitter.
        
        Args:
            samples_per_file: Số samples mỗi file
            min_line_length: Minimum length của mỗi line
        """
        self.samples_per_file = samples_per_file
        self.min_line_length = min_line_length
        self.processed_lines = set()  # Track để tránh duplicate
        
    def read_corpus_lines(self, file_path: str) -> Iterator[str]:
        """
        Read lines từ corpus file với validation.
        
        Args:
            file_path: Path to corpus file
            
        Yields:
            Valid Vietnamese lines
        """
        logger.info(f"Reading corpus: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                valid_count = 0
                
                for line in f:
                    line_count += 1
                    line = line.strip()
                    
                    # Skip empty or too short lines
                    if len(line) < self.min_line_length:
                        continue
                    
                    # Validate Vietnamese text
                    if not is_valid_vietnamese_text(line):
                        continue
                    
                    # Normalize
                    normalized = normalize_vietnamese_text(line)
                    
                    # Skip if already processed (avoid duplicates)
                    if normalized in self.processed_lines:
                        continue
                    
                    self.processed_lines.add(normalized)
                    valid_count += 1
                    yield normalized
                    
                    if line_count % 100000 == 0:
                        logger.info(f"Processed {line_count:,} lines, valid: {valid_count:,}")
        
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    def collect_all_lines(self, corpus_files: List[str]) -> List[str]:
        """
        Collect tất cả valid lines từ corpus files.
        
        Args:
            corpus_files: List of corpus file paths
            
        Returns:
            List of valid unique lines
        """
        all_lines = []
        
        for corpus_file in corpus_files:
            if not os.path.exists(corpus_file):
                logger.warning(f"Corpus file not found: {corpus_file}")
                continue
            
            lines_from_file = list(self.read_corpus_lines(corpus_file))
            all_lines.extend(lines_from_file)
            
            logger.info(f"Collected {len(lines_from_file):,} lines from {corpus_file}")
        
        logger.info(f"Total collected: {len(all_lines):,} unique lines")
        return all_lines
    
    def shuffle_and_split(self, all_lines: List[str], output_dir: str) -> List[str]:
        """
        Shuffle và split lines thành sample files.
        
        Args:
            all_lines: All collected lines
            output_dir: Output directory cho sample files
            
        Returns:
            List of created sample file paths
        """
        # Shuffle to ensure randomness
        logger.info("Shuffling lines...")
        random.shuffle(all_lines)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Split into files
        created_files = []
        total_files = (len(all_lines) + self.samples_per_file - 1) // self.samples_per_file
        
        logger.info(f"Creating {total_files} sample files với {self.samples_per_file:,} samples each")
        
        for i in range(0, len(all_lines), self.samples_per_file):
            file_index = i // self.samples_per_file + 1
            chunk = all_lines[i:i + self.samples_per_file]
            
            # Create filename
            output_file = os.path.join(output_dir, f"training_samples_{file_index:03d}.txt")
            
            # Write chunk to file
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in chunk:
                    f.write(line + '\n')
            
            created_files.append(output_file)
            logger.info(f"Created {output_file} với {len(chunk):,} samples")
        
        return created_files
    
    def create_training_pairs_from_file(self, sample_file: str, output_file: str) -> int:
        """
        Tạo training pairs từ một sample file.
        
        Args:
            sample_file: Input sample file
            output_file: Output JSON file
            
        Returns:
            Number of training pairs created
        """
        from train_atcn import create_training_data_from_corpus
        
        logger.info(f"Creating training pairs from {sample_file}")
        
        # Use existing function
        create_training_data_from_corpus(
            corpus_file=sample_file,
            output_file=output_file,
            max_samples=self.samples_per_file
        )
        
        # Count pairs
        if os.path.exists(output_file):
            import json
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return len(data)
        
        return 0
    
    def split_corpus(self, corpus_files: List[str], output_dir: str = "processed_data/samples") -> List[str]:
        """
        Main function để split corpus thành sample files.
        
        Args:
            corpus_files: List of corpus files
            output_dir: Output directory
            
        Returns:
            List of created sample files
        """
        logger.info("Starting corpus splitting...")
        logger.info(f"Target: {self.samples_per_file:,} samples per file")
        
        # Step 1: Collect all lines
        all_lines = self.collect_all_lines(corpus_files)
        
        if not all_lines:
            logger.error("No valid lines collected from corpus files")
            return []
        
        # Step 2: Shuffle and split
        sample_files = self.shuffle_and_split(all_lines, output_dir)
        
        logger.info(f"Successfully created {len(sample_files)} sample files")
        return sample_files


def create_sample_manifest(sample_files: List[str], manifest_file: str):
    """
    Tạo manifest file listing tất cả sample files.
    
    Args:
        sample_files: List of sample file paths
        manifest_file: Path to manifest file
    """
    import json
    
    manifest = {
        "total_files": len(sample_files),
        "samples_per_file": 500000,
        "created_at": str(os.path.getctime(sample_files[0]) if sample_files else ""),
        "sample_files": [os.path.basename(f) for f in sample_files]
    }
    
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created manifest: {manifest_file}")


def main():
    """Main function để run corpus splitting."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split large corpus into sample files')
    parser.add_argument('--corpus_files', nargs='+', 
                       default=['data/cleaned_comments.txt', 'data/corpus-full.txt'],
                       help='Corpus files to split')
    parser.add_argument('--output_dir', default='processed_data/samples',
                       help='Output directory for sample files')
    parser.add_argument('--samples_per_file', type=int, default=500000,
                       help='Number of samples per file')
    parser.add_argument('--min_line_length', type=int, default=10,
                       help='Minimum line length')
    parser.add_argument('--create_training_data', action='store_true',
                       help='Also create training data JSONs')
    
    args = parser.parse_args()
    
    # Initialize splitter
    splitter = CorpusSplitter(
        samples_per_file=args.samples_per_file,
        min_line_length=args.min_line_length
    )
    
    # Split corpus
    sample_files = splitter.split_corpus(args.corpus_files, args.output_dir)
    
    if not sample_files:
        logger.error("No sample files created")
        return
    
    # Create manifest
    manifest_file = os.path.join(args.output_dir, "sample_manifest.json")
    create_sample_manifest(sample_files, manifest_file)
    
    # Optionally create training data
    if args.create_training_data:
        logger.info("Creating training data from sample files...")
        
        training_dir = os.path.join(args.output_dir, "training_data")
        os.makedirs(training_dir, exist_ok=True)
        
        total_pairs = 0
        for i, sample_file in enumerate(sample_files, 1):
            training_file = os.path.join(training_dir, f"training_data_{i:03d}.json")
            pairs_count = splitter.create_training_pairs_from_file(sample_file, training_file)
            total_pairs += pairs_count
            
            logger.info(f"Sample {i}/{len(sample_files)}: {pairs_count:,} training pairs")
        
        logger.info(f"Total training pairs created: {total_pairs:,}")
    
    logger.info("Corpus splitting completed!")
    logger.info(f"Sample files: {len(sample_files)}")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main() 