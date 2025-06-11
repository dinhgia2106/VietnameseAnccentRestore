#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test để kiểm tra model
"""

import torch
import torch.nn.functional as F
from neural_model import NeuralToneModel

def simple_char_test():
    """Test từng character một cách đơn giản"""
    
    print("SIMPLE CHARACTER TEST")
    print("=" * 40)
    
    model = NeuralToneModel(lightweight=True)
    success = model.load_model(mode="sentence-level")
    
    if not success:
        print("Không load được model!")
        return
    
    model.model.eval()
    
    test_word = "xin"
    print(f"Testing word: '{test_word}'")
    
    # Tạo input tensor
    indices = [model.char_to_idx_input.get(c, 1) for c in test_word]
    while len(indices) < 64:
        indices.append(0)
    
    input_tensor = torch.tensor([indices]).to(model.device)
    
    with torch.no_grad():
        outputs = model.model(input_tensor)
        probs = F.softmax(outputs, dim=-1)
    
    print(f"Input indices: {indices[:len(test_word)]}")
    
    # Lấy character tốt nhất cho mỗi position
    simple_result = ""
    for pos in range(len(test_word)):
        best_idx = torch.argmax(probs[0, pos]).item()
        best_char = model.idx_to_char_output.get(best_idx, '<UNK>')
        prob = probs[0, pos, best_idx].item()
        
        print(f"Position {pos}: '{test_word[pos]}' -> '{best_char}' (prob: {prob:.4f})")
        simple_result += best_char
    
    print(f"Simple result: '{test_word}' -> '{simple_result}'")
    
    # So sánh với predict_word
    print(f"\nComparing with predict_word:")
    beam_results = model.predict_word(test_word, top_k=3)
    for i, (result, score) in enumerate(beam_results):
        print(f"  {i+1}. '{result}' -> {score:.4f}")

def test_different_approach():
    """Test với approach khác"""
    
    print(f"\n{'='*40}")
    print("TESTING DIFFERENT APPROACH")
    print(f"{'='*40}")
    
    model = NeuralToneModel(lightweight=True)
    model.load_model(mode="sentence-level")
    model.model.eval()
    
    test_word = "xin"
    
    # Method 1: Greedy decoding (chọn best char mỗi position)
    indices = [model.char_to_idx_input.get(c, 1) for c in test_word]
    while len(indices) < 64:
        indices.append(0)
    
    input_tensor = torch.tensor([indices]).to(model.device)
    
    with torch.no_grad():
        outputs = model.model(input_tensor)
        probs = F.softmax(outputs, dim=-1)
    
    greedy_result = ""
    for pos in range(len(test_word)):
        best_idx = torch.argmax(probs[0, pos]).item()
        best_char = model.idx_to_char_output.get(best_idx, '<UNK>')
        greedy_result += best_char
    
    print(f"Greedy decoding: '{test_word}' -> '{greedy_result}'")
    
    # Method 2: Check top-3 cho mỗi position
    print(f"\nTop-3 for each position:")
    for pos in range(len(test_word)):
        top_3_probs, top_3_indices = torch.topk(probs[0, pos], 3)
        print(f"Position {pos} ('{test_word[pos]}'):")
        for i, (prob, idx) in enumerate(zip(top_3_probs, top_3_indices)):
            char = model.idx_to_char_output.get(idx.item(), '<UNK>')
            print(f"  {i+1}. '{char}' -> {prob.item():.4f}")

def test_with_toned_examples():
    """Test với examples có sẵn dấu"""
    
    print(f"\n{'='*40}")
    print("TESTING WITH TONED EXAMPLES")
    print(f"{'='*40}")
    
    model = NeuralToneModel(lightweight=True)
    model.load_model(mode="sentence-level")
    
    # Test với các từ có dấu
    toned_examples = ["xin", "chào", "bạn", "tôi"]
    
    for word in toned_examples:
        no_tone = model.remove_tones(word)
        if no_tone != word:
            print(f"\nTesting: '{no_tone}' -> should be '{word}'")
            
            # Greedy approach
            model.model.eval()
            indices = [model.char_to_idx_input.get(c, 1) for c in no_tone]
            while len(indices) < 64:
                indices.append(0)
            
            input_tensor = torch.tensor([indices]).to(model.device)
            
            with torch.no_grad():
                outputs = model.model(input_tensor)
                probs = F.softmax(outputs, dim=-1)
            
            greedy_result = ""
            for pos in range(len(no_tone)):
                best_idx = torch.argmax(probs[0, pos]).item()
                best_char = model.idx_to_char_output.get(best_idx, '<UNK>')
                greedy_result += best_char
            
            print(f"  Greedy: '{greedy_result}'")
            
            # Beam search approach
            beam_results = model.predict_word(no_tone, top_k=1)
            if beam_results:
                print(f"  Beam: '{beam_results[0][0]}'")
            
            # Check if correct
            correct = greedy_result == word
            print(f"  Correct: {correct}")

if __name__ == "__main__":
    simple_char_test()
    test_different_approach() 
    test_with_toned_examples() 