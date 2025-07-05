#!/usr/bin/env python3
"""
Main training script for A-TCN Vietnamese Accent Restoration
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training import ATCNTrainingPipeline, TrainingConfig, SMALL_CONFIG, LARGE_CONFIG, DEFAULT_CONFIG


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train A-TCN model for Vietnamese Accent Restoration"
    )
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=128,
                      help='Embedding dimension (default: 128)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension (default: 256)')
    parser.add_argument('--num_layers', type=int, default=6,
                      help='Number of TCN layers (default: 6)')
    parser.add_argument('--kernel_size', type=int, default=3,
                      help='Kernel size for convolutions (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate (default: 0.1)')
    parser.add_argument('--max_dilation', type=int, default=32,
                      help='Maximum dilation (default: 32)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay (default: 1e-4)')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of epochs (default: 50)')
    parser.add_argument('--max_length', type=int, default=256,
                      help='Maximum sequence length (default: 256)')
    
    # Data parameters
    parser.add_argument('--corpus_dir', type=str, default='corpus_splitted',
                      help='Corpus directory (default: corpus_splitted)')
    parser.add_argument('--vocab_path', type=str, default='vietnamese_char_vocab.json',
                      help='Vocabulary file path (default: vietnamese_char_vocab.json)')
    parser.add_argument('--max_files', type=int, default=None,
                      help='Maximum number of files to load (default: None)')
    parser.add_argument('--max_samples_per_file', type=int, default=10000,
                      help='Maximum samples per file (default: 10000)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Output directory (default: outputs)')
    
    # Presets
    parser.add_argument('--config', type=str, choices=['small', 'default', 'large'],
                      default='default', help='Configuration preset (default: default)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto, cuda, cpu) (default: auto)')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("A-TCN Vietnamese Accent Restoration Training")
    print("=" * 50)
    
    # Select base configuration
    if args.config == 'small':
        config = SMALL_CONFIG
        print("Using SMALL configuration")
    elif args.config == 'large':
        config = LARGE_CONFIG
        print("Using LARGE configuration")
    else:
        config = DEFAULT_CONFIG
        print("Using DEFAULT configuration")
    
    # Override with command line arguments
    config.embedding_dim = args.embedding_dim
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.num_layers
    config.kernel_size = args.kernel_size
    config.dropout = args.dropout
    config.max_dilation = args.max_dilation
    
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.num_epochs = args.num_epochs
    config.max_length = args.max_length
    
    config.corpus_dir = args.corpus_dir
    config.vocab_path = args.vocab_path
    config.max_files = args.max_files
    config.max_samples_per_file = args.max_samples_per_file
    
    config.output_dir = args.output_dir
    
    if args.device != 'auto':
        config.device = args.device
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Model: {config.embedding_dim}d embedding, {config.hidden_dim}d hidden, {config.num_layers} layers")
    print(f"  Training: {config.batch_size} batch size, {config.learning_rate} lr, {config.num_epochs} epochs")
    print(f"  Data: {config.corpus_dir}, max_files={config.max_files}")
    print(f"  Output: {config.output_dir}")
    print(f"  Device: {config.device}")
    
    # Create and run training pipeline
    try:
        pipeline = ATCNTrainingPipeline(config)
        history = pipeline.run()
        
        print("\nTraining completed successfully!")
        print(f"Best validation loss: {pipeline.best_val_loss:.4f}")
        print(f"Total epochs: {pipeline.epoch + 1}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 