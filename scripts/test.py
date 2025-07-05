#!/usr/bin/env python3
"""
Test script for A-TCN Vietnamese Accent Restoration
"""

import argparse
import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import ATCN, ATCNTrainer
from data import VietnameseCharProcessor
from training import TrainingConfig


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config and vocab info
    config_dict = checkpoint.get('config', {})
    vocab_info = checkpoint.get('char_processor_vocab', {})
    
    # Create character processor
    char_processor = VietnameseCharProcessor()
    if vocab_info:
        char_processor.char_to_idx = vocab_info['char_to_idx']
        char_processor.idx_to_char = {int(k): v for k, v in vocab_info['idx_to_char'].items()}
        char_processor.vocab_size = vocab_info['vocab_size']
        char_processor.pad_idx = char_processor.char_to_idx.get('<PAD>', 0)
    
    # Extract model config
    model_config = config_dict.get('model', {})
    vocab_size = char_processor.vocab_size
    
    # Create model
    model = ATCN(
        vocab_size=vocab_size,
        embedding_dim=model_config.get('embedding_dim', 128),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 6),
        kernel_size=model_config.get('kernel_size', 3),
        dropout=model_config.get('dropout', 0.1),
        max_dilation=model_config.get('max_dilation', 32)
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create trainer
    trainer = ATCNTrainer(model, device)
    
    print(f"Model loaded successfully!")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, trainer, char_processor


def test_model_predictions(trainer: ATCNTrainer, char_processor: VietnameseCharProcessor,
                         test_inputs: list, max_length: int = 256):
    """Test model predictions on sample inputs"""
    print(f"\nTesting model predictions:")
    print("=" * 50)
    
    for i, input_text in enumerate(test_inputs, 1):
        # Tokenize input
        tokens = char_processor.tokenize_text(input_text, max_length)
        input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0)  # Add batch dim
        attention_mask = torch.tensor(tokens['attention_mask']).unsqueeze(0)
        
        # Get prediction
        predictions = trainer.predict(input_ids, attention_mask)
        predicted_text = char_processor.indices_to_text(predictions[0].cpu().tolist())
        
        print(f"Test {i}:")
        print(f"  Input:      '{input_text}'")
        print(f"  Prediction: '{predicted_text}'")
        print()


def interactive_test(trainer: ATCNTrainer, char_processor: VietnameseCharProcessor, 
                    max_length: int = 256):
    """Interactive testing mode"""
    print(f"\nInteractive testing mode (type 'quit' to exit):")
    print("=" * 50)
    
    while True:
        try:
            input_text = input("Nhap text khong dau: ").strip()
            
            if input_text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not input_text:
                continue
            
            # Tokenize and predict
            tokens = char_processor.tokenize_text(input_text, max_length)
            input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0)
            attention_mask = torch.tensor(tokens['attention_mask']).unsqueeze(0)
            
            predictions = trainer.predict(input_ids, attention_mask)
            predicted_text = char_processor.indices_to_text(predictions[0].cpu().tolist())
            
            print(f"Ket qua: '{predicted_text}'")
            print()
            
        except KeyboardInterrupt:
            break
    
    print("Exiting interactive mode.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test A-TCN model for Vietnamese Accent Restoration"
    )
    
    parser.add_argument('checkpoint_path', type=str,
                      help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto, cuda, cpu) (default: auto)')
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode')
    parser.add_argument('--test_inputs', nargs='*',
                      default=[
                          "toi yeu viet nam",
                          "chung ta la mot",
                          "hoc tap cham chi",
                          "co gai xinh dep",
                          "dem nay troi rat dep"
                      ],
                      help='Test input texts')
    parser.add_argument('--max_length', type=int, default=256,
                      help='Maximum sequence length (default: 256)')
    
    return parser.parse_args()


def main():
    """Main testing function"""
    args = parse_args()
    
    print("A-TCN Vietnamese Accent Restoration Testing")
    print("=" * 50)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    try:
        model, trainer, char_processor = load_model_from_checkpoint(
            args.checkpoint_path, device
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test predictions
    if not args.interactive:
        test_model_predictions(
            trainer, char_processor, args.test_inputs, args.max_length
        )
    
    # Interactive mode
    if args.interactive:
        interactive_test(trainer, char_processor, args.max_length)


if __name__ == "__main__":
    main() 