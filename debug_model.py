#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script để kiểm tra tại sao model cho kết quả sai
"""

import torch
import numpy as np
from neural_model import NeuralToneModel

def debug_model_inference():
    """Debug việc inference của model"""
    
    print("DEBUG MODEL INFERENCE")
    print("=" * 50)
    
    # Load model
    model = NeuralToneModel(lightweight=True)
    success = model.load_model(mode="sentence-level")
    
    if not success:
        print("❌ Không thể load model!")
        return
    
    print("✓ Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Debug vocabulary
    print(f"\nVOCABULARY DEBUG:")
    print(f"Input vocab size: {len(model.char_to_idx_input)}")
    print(f"Output vocab size: {len(model.char_to_idx_output)}")
    
    print(f"\nSample input vocab (first 20):")
    for i, (char, idx) in enumerate(list(model.char_to_idx_input.items())[:20]):
        print(f"  '{char}' -> {idx}")
    
    print(f"\nSample output vocab (first 20):")
    for i, (char, idx) in enumerate(list(model.char_to_idx_output.items())[:20]):
        print(f"  '{char}' -> {idx}")
    
    # Test simple character mapping
    test_word = "xin"
    print(f"\nTEST CHARACTER MAPPING for '{test_word}':")
    
    input_chars = list(test_word)
    input_indices = [model.char_to_idx_input.get(c, 1) for c in input_chars]
    print(f"Input chars: {input_chars}")
    print(f"Input indices: {input_indices}")
    
    # Check if characters exist in vocab
    for char in input_chars:
        if char in model.char_to_idx_input:
            print(f"  '{char}' -> {model.char_to_idx_input[char]} ✓")
        else:
            print(f"  '{char}' -> UNK (1) ❌")
    
    # Test model inference step by step
    print(f"\nSTEP-BY-STEP INFERENCE:")
    
    model.model.eval()
    
    # Prepare input như trong predict_word
    indices = [model.char_to_idx_input.get(c, 1) for c in test_word[:64]]
    while len(indices) < 64:
        indices.append(0)  # PAD
    
    input_tensor = torch.tensor([indices]).to(model.device)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor (first 10): {input_tensor[0][:10].tolist()}")
    
    with torch.no_grad():
        outputs = model.model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=-1)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Probabilities shape: {probs.shape}")
    
    # Check predictions for each position
    for pos in range(len(test_word)):
        pos_probs = probs[0, pos]
        top_5_probs, top_5_indices = torch.topk(pos_probs, 5)
        
        print(f"\nPosition {pos} (char '{test_word[pos]}'):")
        for i, (prob, idx) in enumerate(zip(top_5_probs, top_5_indices)):
            char = model.idx_to_char_output.get(idx.item(), '<UNK>')
            print(f"  {i+1}. '{char}' (idx:{idx.item()}) -> {prob.item():.4f}")
    
    # Test full prediction
    print(f"\nFULL PREDICTION TEST:")
    results = model.predict_word(test_word, top_k=5)
    for i, (predicted, score) in enumerate(results):
        print(f"  {i+1}. '{predicted}' -> {score:.4f}")

def check_training_data_compatibility():
    """Kiểm tra tính tương thích của training data"""
    
    print(f"\n{'='*50}")
    print("TRAINING DATA COMPATIBILITY CHECK")
    print(f"{'='*50}")
    
    model = NeuralToneModel(lightweight=True)
    
    # Test remove_tones function
    test_cases = [
        "xin", "chào", "bạn", "tôi", "là", "sinh", "viên"
    ]
    
    print("Testing tone removal:")
    for word in test_cases:
        no_tone = model.remove_tones(word)
        print(f"  '{word}' -> '{no_tone}'")
    
    # Test if có data files
    import os
    data_files = ["data/Viet74K_clean.txt", "data/cleaned_comments.txt"]
    
    print(f"\nData files check:")
    for file_path in data_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ✓ {file_path} exists ({file_size:.2f} MB)")
            
            # Read a few lines
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [f.readline().strip() for _ in range(3)]
            print(f"    Sample lines: {lines}")
        else:
            print(f"  ❌ {file_path} not found")

def suggest_fixes():
    """Đề xuất cách fix"""
    
    print(f"\n{'='*50}")
    print("SUGGESTED FIXES")
    print(f"{'='*50}")
    
    print("Có thể nguyên nhân và cách fix:")
    
    print("\n1. MODEL CHƯA ĐƯỢC TRAIN ĐỦ:")
    print("   - Model hiện tại chỉ train 2 epochs")
    print("   - Cần train nhiều epochs hơn để học tốt")
    print("   - Chạy: python neural_model.py --sentences --lightweight --train")
    
    print("\n2. VOCABULARY MISMATCH:")
    print("   - Có thể vocab trong checkpoint không khớp với data hiện tại")
    print("   - Train lại từ đầu để đảm bảo consistency")
    
    print("\n3. DATA QUALITY:")
    print("   - Kiểm tra quality của training data")
    print("   - Đảm bảo có đủ examples cho tone restoration")
    
    print("\n4. MODEL ARCHITECTURE:")
    print("   - Model quá nhỏ (embedding=32, hidden=64)")
    print("   - Thử với model lớn hơn: python neural_model.py --sentences --train")
    
    print("\n5. LEARNING RATE / TRAINING PARAMS:")
    print("   - Có thể learning rate không phù hợp")
    print("   - Model converge sai local minimum")

def main():
    debug_model_inference()
    check_training_data_compatibility()
    suggest_fixes()

if __name__ == "__main__":
    main() 