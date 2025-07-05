#!/usr/bin/env python3
"""
Integration test for unified codebase
"""

import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from models import ATCN, ATCNTrainer
        print("  ✓ Models imported successfully")
        
        from data import VietnameseCharProcessor, VietnameseCharDataset, create_data_loaders
        print("  ✓ Data modules imported successfully")
        
        from training import ATCNTrainingPipeline, TrainingConfig, SMALL_CONFIG
        print("  ✓ Training modules imported successfully")
        
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_character_processor():
    """Test character processor"""
    print("\nTesting Character Processor...")
    
    try:
        from data import VietnameseCharProcessor
        
        # Test initialization
        processor = VietnameseCharProcessor()
        assert processor.vocab_size > 200, "Vocab size should be > 200"
        print(f"  ✓ Processor initialized: vocab_size={processor.vocab_size}")
        
        # Test tokenization
        text = "toi yeu viet nam"
        tokens = processor.tokenize_text(text, max_length=32)
        assert 'input_ids' in tokens, "Should have input_ids"
        assert 'attention_mask' in tokens, "Should have attention_mask"
        print(f"  ✓ Tokenization working")
        
        # Test reconstruction
        reconstructed = processor.indices_to_text(tokens['input_ids'])
        assert reconstructed == text, f"Reconstruction failed: {reconstructed} != {text}"
        print(f"  ✓ Reconstruction working")
        
        return True
    except Exception as e:
        print(f"  ✗ Character processor test failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting Model Creation...")
    
    try:
        from models import ATCN, ATCNTrainer
        from data import VietnameseCharProcessor
        
        # Create processor and model
        processor = VietnameseCharProcessor()
        model = ATCN(
            vocab_size=processor.vocab_size,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=3,
            kernel_size=3
        )
        
        print(f"  ✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test trainer
        trainer = ATCNTrainer(model, 'cpu')
        print(f"  ✓ Trainer created")
        
        return True
    except Exception as e:
        print(f"  ✗ Model creation test failed: {e}")
        return False

def test_training_config():
    """Test training configuration"""
    print("\nTesting Training Configuration...")
    
    try:
        from training import TrainingConfig, SMALL_CONFIG
        
        # Test default config
        config = TrainingConfig()
        config.validate()
        print(f"  ✓ Default config valid")
        
        # Test small config
        SMALL_CONFIG.validate()
        print(f"  ✓ Small config valid")
        
        # Test config serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict), "Config should serialize to dict"
        print(f"  ✓ Config serialization working")
        
        return True
    except Exception as e:
        print(f"  ✗ Training config test failed: {e}")
        return False

def test_end_to_end():
    """Test minimal end-to-end workflow"""
    print("\nTesting End-to-End Workflow...")
    
    try:
        import torch
        from models import ATCN, ATCNTrainer
        from data import VietnameseCharProcessor
        from training import TrainingConfig
        
        # Create minimal setup
        processor = VietnameseCharProcessor()
        model = ATCN(
            vocab_size=processor.vocab_size,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2
        )
        trainer = ATCNTrainer(model, 'cpu')
        
        # Create fake batch
        batch_size = 2
        seq_len = 16
        batch = {
            'input_ids': torch.randint(4, processor.vocab_size, (batch_size, seq_len)),
            'target_ids': torch.randint(4, processor.vocab_size, (batch_size, seq_len)),
            'input_attention_mask': torch.ones(batch_size, seq_len),
            'target_attention_mask': torch.ones(batch_size, seq_len)
        }
        
        # Test forward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = trainer.train_step(batch, optimizer)
        assert 'loss' in metrics, "Should return loss"
        assert 'accuracy' in metrics, "Should return accuracy"
        print(f"  ✓ Training step: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
        
        # Test validation
        val_metrics = trainer.val_step(batch)
        print(f"  ✓ Validation step: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}")
        
        # Test prediction
        predictions = trainer.predict(batch['input_ids'], batch['input_attention_mask'])
        assert predictions.shape == batch['input_ids'].shape, "Prediction shape mismatch"
        print(f"  ✓ Prediction working")
        
        return True
    except Exception as e:
        print(f"  ✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("UNIFIED CODEBASE INTEGRATION TEST")
    print("="*60)
    
    tests = [
        test_imports,
        test_character_processor,
        test_model_creation,
        test_training_config,
        test_end_to_end
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - Codebase is unified and working!")
        return True
    else:
        print(f"✗ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 