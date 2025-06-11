#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo cải tiến Neural Tone Restoration Model
So sánh hiệu suất giữa các phiên bản khác nhau
"""

import time
import os
import sys
from neural_model import NeuralToneModel

def compare_models():
    """So sánh hiệu suất các phiên bản model"""
    
    print("DEMO CẢI TIẾN NEURAL TONE RESTORATION")
    print("=" * 60)
    
    # Test cases với các trường hợp khó
    test_cases = [
        # Trường hợp cơ bản
        "xin chao ban",
        "toi la sinh vien",
        "cam on ban rat nhieu",
        
        # Trường hợp cần ngữ cảnh câu
        "anh ay thich hop",  # "hợp" vs "hộp"
        "co ay rat thich hop",  # ngữ cảnh rõ hơn
        "toi mua mot cai hop",  # ngữ cảnh rõ hơn cho "hộp"
        
        # Câu dài
        "hom nay troi rat dep va trong xanh",
        "chung toi se di du lich vao cuoi tuan",
        
        # Trường hợp phức tạp
        "ban co the giup toi khong",
        "rat vui duoc gap ban o day"
    ]
    
    print("Test cases:")
    for i, case in enumerate(test_cases, 1):
        print(f"  {i:2d}. {case}")
    print()
    
    # Test các phiên bản model
    model_configs = [
        {
            'name': 'Word-level Standard',
            'lightweight': False,
            'sentence_level': False,
            'description': 'Model gốc xử lý từng từ riêng lẻ'
        },
        {
            'name': 'Word-level Lightweight', 
            'lightweight': True,
            'sentence_level': False,
            'description': 'Model nhẹ xử lý từng từ'
        },
        {
            'name': 'Sentence-level Standard',
            'lightweight': False, 
            'sentence_level': True,
            'description': 'Model xử lý toàn bộ câu với ngữ cảnh'
        },
        {
            'name': 'Sentence-level Lightweight',
            'lightweight': True,
            'sentence_level': True, 
            'description': 'Model nhẹ xử lý toàn bộ câu'
        }
    ]
    
    results = {}
    
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"TESTING: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        try:
            # Khởi tạo model
            model = NeuralToneModel(lightweight=config['lightweight'])
            
            # Thử load model
            mode = "sentence-level" if config['sentence_level'] else "word-level"
            loaded = model.load_model(mode=mode)
            
            if not loaded:
                print(f"⚠️  Model {config['name']} chưa được train!")
                print("   Để train model này, chạy:")
                if config['sentence_level'] and config['lightweight']:
                    print("   python neural_model.py --sentences --lightweight --train")
                elif config['sentence_level']:
                    print("   python neural_model.py --sentences --train")
                elif config['lightweight']:
                    print("   python neural_model.py --lightweight --train")
                else:
                    print("   python neural_model.py --train")
                continue
            
            # Test performance
            model._trained_sentence_level = config['sentence_level']
            
            total_time = 0
            correct_predictions = 0
            model_results = []
            
            print(f"\nTesting {len(test_cases)} cases...")
            
            for i, test_case in enumerate(test_cases):
                start_time = time.time()
                
                # Dự đoán
                if config['sentence_level']:
                    predictions = model.restore_tones_optimized(
                        test_case, max_results=3, use_sentence_level=True
                    )
                else:
                    predictions = model.restore_tones_optimized(
                        test_case, max_results=3, use_sentence_level=False  
                    )
                
                end_time = time.time()
                processing_time = (end_time - start_time) * 1000
                total_time += processing_time
                
                best_prediction = predictions[0][0] if predictions else test_case
                confidence = predictions[0][1] if predictions else 0.0
                
                model_results.append({
                    'input': test_case,
                    'output': best_prediction,
                    'confidence': confidence,
                    'time_ms': processing_time,
                    'alternatives': len(predictions)
                })
                
                print(f"  {i+1:2d}. {test_case:30} -> {best_prediction:30} "
                      f"({confidence:.3f}, {processing_time:5.1f}ms)")
            
            # Tính toán metrics
            avg_time = total_time / len(test_cases)
            
            if model.model:
                param_count = sum(p.numel() for p in model.model.parameters())
                model_size_mb = param_count * 4 / (1024 * 1024)
            else:
                param_count = 0
                model_size_mb = 0
            
            # Lưu kết quả
            results[config['name']] = {
                'avg_time_ms': avg_time,
                'total_time_ms': total_time,
                'param_count': param_count,
                'model_size_mb': model_size_mb,
                'predictions': model_results,
                'loaded': True
            }
            
            print(f"\nPerformance Summary:")
            print(f"  Average time: {avg_time:.2f}ms")
            print(f"  Total time: {total_time:.2f}ms") 
            print(f"  Parameters: {param_count:,}")
            print(f"  Model size: {model_size_mb:.2f}MB")
            
        except Exception as e:
            print(f" Error testing {config['name']}: {e}")
            results[config['name']] = {'loaded': False, 'error': str(e)}
    
    # So sánh kết quả
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Model':<25} {'Avg Time':<12} {'Parameters':<12} {'Size (MB)':<10} {'Status'}")
    print("-" * 80)
    
    for name, result in results.items():
        if result.get('loaded', False):
            print(f"{name:<25} {result['avg_time_ms']:>8.2f}ms "
                  f"{result['param_count']:>10,} {result['model_size_mb']:>8.2f} "
                  f"{'✓ Ready'}")
        else:
            print(f"{name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'✗ Not trained'}")
    
    # Detailed comparison cho những model đã train
    trained_models = {k: v for k, v in results.items() if v.get('loaded', False)}
    
    if len(trained_models) >= 2:
        print(f"\n{'='*60}")
        print("DETAILED COMPARISON")
        print(f"{'='*60}")
        
        # Speed comparison
        fastest_time = min(r['avg_time_ms'] for r in trained_models.values())
        print("\nSpeed Performance (vs fastest):")
        for name, result in trained_models.items():
            ratio = result['avg_time_ms'] / fastest_time
            print(f"  {name:<25}: {result['avg_time_ms']:6.2f}ms ({ratio:.1f}x)")
        
        # Size comparison  
        smallest_size = min(r['model_size_mb'] for r in trained_models.values())
        print(f"\nModel Size (vs smallest):")
        for name, result in trained_models.items():
            ratio = result['model_size_mb'] / smallest_size if smallest_size > 0 else 1
            print(f"  {name:<25}: {result['model_size_mb']:6.2f}MB ({ratio:.1f}x)")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")  
    print(f"{'='*60}")
    
    print("1. Để có độ chính xác cao nhất với ngữ cảnh câu:")
    print("   → Sử dụng Sentence-level Standard")
    print("   → python neural_model.py --sentences")
    
    print("\n2. Để có tốc độ nhanh nhất:")
    print("   → Sử dụng Word-level Lightweight") 
    print("   → python neural_model.py --lightweight")
    
    print("\n3. Cân bằng giữa độ chính xác và tốc độ:")
    print("   → Sử dụng Sentence-level Lightweight")
    print("   → python neural_model.py --sentences --lightweight")
    
    print("\n4. Để triển khai trên thiết bị hạn chế tài nguyên:")
    print("   → Ưu tiên các phiên bản Lightweight")
    print("   → Kích thước model nhỏ hơn đáng kể")

def demo_context_examples():
    """Demo các ví dụ cần ngữ cảnh câu"""
    print(f"\n{'='*60}")
    print("DEMO: TẦM QUAN TRỌNG CỦA NGỮ CẢNH CÂU")
    print(f"{'='*60}")
    
    # Các ví dụ cần ngữ cảnh
    context_examples = [
        {
            'input': 'hop',
            'contexts': [
                'anh ay thich hop',      # Should be "hợp" (suitable)
                'toi mua mot cai hop',   # Should be "hộp" (box)
                'hop bao nhieu tien'     # Could be either
            ]
        },
        {
            'input': 'ban',
            'contexts': [
                'ban toi la sinh vien',  # "bạn" (friend)
                'toi ngoi tren ban',     # "bàn" (table)  
                'ban sach nay rat hay'   # "bản" (version/copy)
            ]
        },
        {
            'input': 'may',
            'contexts': [
                'may gio roi',           # "mấy" (what time)
                'may tinh cua toi',      # "máy" (machine)
                'troi may',              # "mây" (clouds)
                'may quan ao'            # "may" (sew)
            ]
        }
    ]
    
    print("Những trường hợp này cho thấy tại sao sentence-level processing quan trọng:")
    print("Cùng một từ có thể có nghĩa và dấu khác nhau tùy vào ngữ cảnh.\n")
    
    for example in context_examples:
        print(f"Từ '{example['input']}' trong các ngữ cảnh khác nhau:")
        for context in example['contexts']:
            print(f"  • {context}")
        print()

def main():
    """Main function"""
    print("Bắt đầu demo cải tiến Neural Tone Restoration...")
    
    # Demo ví dụ ngữ cảnh
    demo_context_examples()
    
    # So sánh các model
    compare_models()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Train các model chưa có bằng cách chạy:")
    print("   python neural_model.py --train                    # Word-level standard")
    print("   python neural_model.py --lightweight --train     # Word-level lightweight")  
    print("   python neural_model.py --sentences --train       # Sentence-level standard")
    print("   python neural_model.py --sentences --lightweight --train  # Sentence-level lightweight")
    
    print("\n2. Test interactive với model đã train:")
    print("   python neural_model.py --sentences               # Sentence-level")
    print("   python neural_model.py --lightweight             # Lightweight")
    
    print("\n3. So sánh chi tiết:")
    print("   python demo_neural_improvements.py")

if __name__ == "__main__":
    main() 