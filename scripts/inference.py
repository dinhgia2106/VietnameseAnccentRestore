#!/usr/bin/env python3
"""
Inference script for A-TCN Vietnamese Accent Restoration
Production-ready script for adding diacritics to Vietnamese text
"""

import argparse
import sys
import os
import json
from typing import List, Dict
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import ATCN, ATCNTrainer
from data import VietnameseCharProcessor


class VietnameseAccentRestorer:
    """Production-ready Vietnamese accent restoration model"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = None
        self.trainer = None
        self.char_processor = None
        self.max_length = 256
        
        self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        print(f"Loading model from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get config and vocab info
        config_dict = checkpoint.get('config', {})
        vocab_info = checkpoint.get('char_processor_vocab', {})
        
        # Create character processor
        self.char_processor = VietnameseCharProcessor()
        if vocab_info:
            self.char_processor.char_to_idx = vocab_info['char_to_idx']
            self.char_processor.idx_to_char = {int(k): v for k, v in vocab_info['idx_to_char'].items()}
            self.char_processor.vocab_size = vocab_info['vocab_size']
            self.char_processor.pad_idx = self.char_processor.char_to_idx.get('<PAD>', 0)
        
        # Extract model config
        model_config = config_dict.get('model', {})
        
        # Create model
        self.model = ATCN(
            vocab_size=self.char_processor.vocab_size,
            embedding_dim=model_config.get('embedding_dim', 128),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 6),
            kernel_size=model_config.get('kernel_size', 3),
            dropout=model_config.get('dropout', 0.1),
            max_dilation=model_config.get('max_dilation', 32)
        )
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create trainer
        self.trainer = ATCNTrainer(self.model, str(self.device))
        
        print(f"Model loaded successfully!")
    
    def restore_accents(self, text: str) -> str:
        """
        Restore accents for Vietnamese text
        
        Args:
            text: Input text without diacritics
            
        Returns:
            Text with restored diacritics
        """
        # Tokenize input
        tokens = self.char_processor.tokenize_text(text, self.max_length)
        input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(tokens['attention_mask']).unsqueeze(0)
        
        # Get prediction using constrained generation
        with torch.no_grad():
            predictions = self.trainer.predict_with_constraints(
                input_ids, self.char_processor, attention_mask, temperature=0.1
            )
            predicted_text = self.char_processor.indices_to_text(predictions[0].cpu().tolist())
        
        return predicted_text
    
    def restore_accents_batch(self, texts: List[str]) -> List[str]:
        """
        Restore accents for multiple texts
        
        Args:
            texts: List of input texts without diacritics
            
        Returns:
            List of texts with restored diacritics
        """
        results = []
        for text in texts:
            restored_text = self.restore_accents(text)
            results.append(restored_text)
        return results


def process_file(restorer: VietnameseAccentRestorer, input_file: str, output_file: str):
    """Process a file of Vietnamese text"""
    print(f"Processing file: {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    results = []
    for line in lines:
        line = line.strip()
        if line:
            restored = restorer.restore_accents(line)
            results.append(restored)
        else:
            results.append('')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result + '\n')
    
    print(f"Processed {len(lines)} lines")


def process_json(restorer: VietnameseAccentRestorer, input_file: str, output_file: str, 
                input_key: str = 'text', output_key: str = 'restored_text'):
    """Process a JSON file with text data"""
    print(f"Processing JSON file: {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        for item in data:
            if input_key in item:
                item[output_key] = restorer.restore_accents(item[input_key])
    elif isinstance(data, dict):
        if input_key in data:
            data[output_key] = restorer.restore_accents(data[input_key])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed JSON data")


def interactive_mode(restorer: VietnameseAccentRestorer):
    """Interactive mode for testing"""
    print("\nInteractive Vietnamese Accent Restoration")
    print("Type Vietnamese text without diacritics, press Enter to get result")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            text = input("\nInput: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            result = restorer.restore_accents(text)
            print(f"Output: {result}")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Vietnamese Accent Restoration Inference"
    )
    
    parser.add_argument('checkpoint_path', type=str,
                      help='Path to model checkpoint')
    
    # Input/Output
    parser.add_argument('--input_file', type=str,
                      help='Input file path')
    parser.add_argument('--output_file', type=str,
                      help='Output file path')
    parser.add_argument('--input_format', type=str, choices=['text', 'json'],
                      default='text', help='Input file format (default: text)')
    
    # JSON specific options
    parser.add_argument('--input_key', type=str, default='text',
                      help='JSON key for input text (default: text)')
    parser.add_argument('--output_key', type=str, default='restored_text',
                      help='JSON key for output text (default: restored_text)')
    
    # Direct text input
    parser.add_argument('--text', type=str,
                      help='Direct text input')
    
    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto, cuda, cpu) (default: auto)')
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()
    
    print("Vietnamese Accent Restoration - Inference")
    print("=" * 50)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    try:
        restorer = VietnameseAccentRestorer(args.checkpoint_path, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Process based on arguments
    if args.interactive:
        interactive_mode(restorer)
    
    elif args.text:
        result = restorer.restore_accents(args.text)
        print(f"\nInput:  {args.text}")
        print(f"Output: {result}")
    
    elif args.input_file and args.output_file:
        if args.input_format == 'json':
            process_json(restorer, args.input_file, args.output_file, 
                        args.input_key, args.output_key)
        else:
            process_file(restorer, args.input_file, args.output_file)
    
    else:
        print("Please specify one of:")
        print("  --interactive: Interactive mode")
        print("  --text: Direct text input")
        print("  --input_file and --output_file: File processing")
        sys.exit(1)


if __name__ == "__main__":
    main() 