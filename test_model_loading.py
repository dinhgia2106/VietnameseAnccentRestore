#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra việc load model sau khi fix
"""

import os
from neural_model import NeuralToneModel

def test_model_loading():
    """Test việc load các model khác nhau"""
    
    print("TESTING MODEL LOADING AFTER FIX")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {'name': 'Standard sentence-level', 'lightweight': False, 'mode': 'sentence-level'},
        {'name': 'Lightweight sentence-level', 'lightweight': True, 'mode': 'sentence-level'},
        {'name': 'Standard word-level', 'lightweight': False, 'mode': 'word-level'},
        {'name': 'Lightweight word-level', 'lightweight': True, 'mode': 'word-level'},
    ]
    
    for config in test_configs:
        print(f"\n{'-'*50}")
        print(f"Testing: {config['name']}")
        print(f"Lightweight: {config['lightweight']}, Mode: {config['mode']}")
        print(f"{'-'*50}")
        
        try:
            # Tạo model instance
            model = NeuralToneModel(lightweight=config['lightweight'])
            
            # Thử load model
            success = model.load_model(mode=config['mode'])
            
            if success:
                print(f"✓ SUCCESS: {config['name']} loaded successfully!")
                
                # Test quick inference
                test_text = "xin chao ban"
                results = model.restore_tones(test_text, max_results=3)
                print(f"  Test inference: '{test_text}' -> '{results[0][0]}' (score: {results[0][1]:.3f})")
                
                # Model info
                if model.model:
                    param_count = sum(p.numel() for p in model.model.parameters())
                    model_size_mb = param_count * 4 / (1024 * 1024)
                    print(f"  Parameters: {param_count:,} (~{model_size_mb:.2f} MB)")
                    print(f"  Sentence-level trained: {getattr(model, '_trained_sentence_level', False)}")
                
            else:
                print(f"✗ FAILED: Could not load {config['name']}")
                print("  This is expected if the model hasn't been trained yet.")
                
        except Exception as e:
            print(f"✗ ERROR: {config['name']} - {e}")
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    # Check what models exist
    model_dir = "models/neural"
    if os.path.exists(model_dir):
        print(f"\nAvailable model files in {model_dir}:")
        for filename in os.listdir(model_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(model_dir, filename)
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {filename} ({file_size:.2f} MB)")
    else:
        print(f"\nNo model directory found at {model_dir}")
    
    print(f"\nTo train missing models:")
    print("  python neural_model.py --train                              # Word-level standard")
    print("  python neural_model.py --lightweight --train               # Word-level lightweight")
    print("  python neural_model.py --sentences --train                 # Sentence-level standard")
    print("  python neural_model.py --sentences --lightweight --train   # Sentence-level lightweight")

def test_auto_detection():
    """Test auto-detection của model configuration"""
    print(f"\n{'='*50}")
    print("TESTING AUTO-DETECTION")
    print(f"{'='*50}")
    
    model_dir = "models/neural"
    if not os.path.exists(model_dir):
        print("No models found for auto-detection test")
        return
    
    # Test với lightweight=False nhưng load lightweight model
    print("\nTest 1: lightweight=False init, load lightweight model")
    model = NeuralToneModel(lightweight=False)  # Init as standard
    
    # Tìm lightweight model
    for filename in os.listdir(model_dir):
        if "sentence-level" in filename and filename.endswith('.pth'):
            filepath = os.path.join(model_dir, filename)
            print(f"Trying to load: {filename}")
            
            # Load trực tiếp file này
            success = model._load_checkpoint(filepath, "sentence-level")
            if success:
                print("✓ Auto-detection successful!")
                print(f"  Original lightweight setting: False")
                print(f"  Detected lightweight: {model.lightweight}")
                break
            else:
                print("✗ Auto-detection failed")
    
    # Test với lightweight=True nhưng load standard model (nếu có)
    print("\nTest 2: lightweight=True init, try to load any available model")
    model2 = NeuralToneModel(lightweight=True)  # Init as lightweight
    success = model2.load_model()
    if success:
        print("✓ Load successful with auto-detection!")
    else:
        print("✗ No compatible model found")

if __name__ == "__main__":
    test_model_loading()
    test_auto_detection() 