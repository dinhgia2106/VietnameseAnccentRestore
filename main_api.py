"""
Vietnamese Neural Tone Restoration API
Hệ thống khôi phục dấu tiếng Việt sử dụng Neural Network (Bi-GRU)
"""

import os
import time
import re
from typing import List, Tuple, Dict, Optional

# Import neural model
from neural_model import NeuralToneModel

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc="", disable=False):
        if desc:
            print(f"{desc}...")
        return iterable

class VietnameseToneAPI:
    """API cho Neural Tone Restoration"""
    
    def __init__(self):
        self.neural_model = NeuralToneModel()
        self.model_loaded = False
        
    def load_model(self) -> bool:
        """Load neural model"""
        try:
            self.model_loaded = self.neural_model.load_model()
            if self.model_loaded:
                print("Đã load Neural Model thành công")
            return self.model_loaded
        except Exception as e:
            print(f"Lỗi load Neural Model: {e}")
            return False
    
    def train_model(self, dict_path: str, corpus_path: str, epochs: int = 15, batch_size: int = 64) -> None:
        """Training neural model"""
        print("Training Neural Model...")
        self.neural_model.train(dict_path, corpus_path, epochs, batch_size)
        self.model_loaded = True
    
    def restore_tones(self, text: str, max_results: int = 5) -> List[Tuple[str, float]]:
        """Khôi phục dấu cho văn bản"""
        if not self.model_loaded:
            print("Model chưa được load!")
            return [(text, 1.0)]
        
        return self.neural_model.restore_tones(text, max_results)
    
    def restore_word(self, word: str, max_results: int = 5) -> List[Tuple[str, float]]:
        """Khôi phục dấu cho một từ"""
        if not self.model_loaded:
            print("Model chưa được load!")
            return [(word, 1.0)]
        
        return self.neural_model.predict_word(word, max_results)
    
    def get_best_result(self, text: str) -> str:
        """Lấy kết quả tốt nhất"""
        results = self.restore_tones(text, max_results=1)
        return results[0][0] if results else text
    
    def benchmark(self, test_texts: List[str], iterations: int = 5) -> Dict:
        """Benchmark neural model"""
        
        if not self.model_loaded:
            print("Model chưa được load!")
            return {}
        
        print(f"Benchmark Neural Model - {iterations} iterations")
        print("=" * 50)
        
        times = []
        total_chars = 0
        
        # Warm up
        for _ in range(3):
            self.restore_tones("test warm up")
        
        # Benchmark
        for iteration in range(iterations):
            for text in test_texts:
                start_time = time.time()
                self.restore_tones(text)
                times.append(time.time() - start_time)
                total_chars += len(text)
        
        # Calculate metrics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput = total_chars / sum(times)
        
        results = {
            'iterations': iterations,
            'total_calls': len(times),
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'total_time_ms': sum(times) * 1000,
            'throughput_chars_per_sec': throughput,
            'total_chars': total_chars
        }
        
        print(f"Tổng số lần gọi: {results['total_calls']}")
        print(f"Thời gian trung bình: {results['avg_time_ms']:.2f}ms")
        print(f"Thời gian min/max: {results['min_time_ms']:.2f}ms / {results['max_time_ms']:.2f}ms")
        print(f"Tốc độ xử lý: {results['throughput_chars_per_sec']:.0f} ký tự/giây")
        
        return results
    
    def process_file(self, input_path: str, output_path: str, max_results: int = 3, 
                    chunk_size: int = 100) -> None:
        """Xử lý file văn bản lớn"""
        
        if not self.model_loaded:
            print("Model chưa được load!")
            return
        
        print(f"Xử lý file: {input_path}")
        print(f"Output: {output_path}")
        print(f"Chunk size: {chunk_size}")
        
        # Đếm tổng số dòng
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        print(f"Tổng số dòng: {total_lines}")
        
        # Tạo thư mục output nếu cần
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        processed_lines = 0
        start_time = time.time()
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            chunk = []
            
            pbar = tqdm(total=total_lines, desc="Processing", disable=not TQDM_AVAILABLE, unit="lines")
            
            for line in infile:
                chunk.append(line.strip())
                
                if len(chunk) >= chunk_size:
                    # Xử lý chunk
                    for text in chunk:
                        if text:
                            results = self.restore_tones(text, max_results)
                            if max_results == 1:
                                outfile.write(f"{results[0][0]}\n")
                            else:
                                outfile.write(f"Input: {text}\n")
                                for i, (restored, score) in enumerate(results, 1):
                                    outfile.write(f"  {i}. {restored} (score: {score:.4f})\n")
                                outfile.write("\n")
                        else:
                            outfile.write("\n")
                    
                    processed_lines += len(chunk)
                    pbar.update(len(chunk))
                    chunk = []
            
            # Xử lý chunk cuối
            if chunk:
                for text in chunk:
                    if text:
                        results = self.restore_tones(text, max_results)
                        if max_results == 1:
                            outfile.write(f"{results[0][0]}\n")
                        else:
                            outfile.write(f"Input: {text}\n")
                            for i, (restored, score) in enumerate(results, 1):
                                outfile.write(f"  {i}. {restored} (score: {score:.4f})\n")
                            outfile.write("\n")
                    else:
                        outfile.write("\n")
                
                processed_lines += len(chunk)
                pbar.update(len(chunk))
            
            pbar.close()
        
        elapsed_time = time.time() - start_time
        
        print(f"\nHoàn thành!")
        print(f"Đã xử lý: {processed_lines} dòng")
        print(f"Thời gian: {elapsed_time:.2f}s")
        print(f"Tốc độ: {processed_lines/elapsed_time:.1f} dòng/giây")
    
    def get_model_info(self) -> Dict:
        """Lấy thông tin model"""
        return {
            'model_type': 'Neural Network (Bi-GRU)',
            'loaded': self.model_loaded,
            'features': [
                'Bidirectional GRU',
                'Character embedding (64D)',
                'Hidden size: 128D', 
                'Sequence-to-sequence labeling',
                'Beam search decoding',
                'PyTorch-based'
            ],
            'strengths': [
                'Độ chính xác cao (~95%+)',
                'Generalize tốt với unseen words',
                'End-to-end learning',
                'Context-aware predictions'
            ],
            'requirements': [
                'PyTorch',
                'GPU recommended for training',
                'CPU inference OK'
            ]
        }

def main():
    """Demo Vietnamese Neural Tone API"""
    print("VIETNAMESE NEURAL TONE RESTORATION API")
    print("Hệ thống khôi phục dấu tiếng Việt sử dụng Neural Network")
    print("=" * 65)
    
    # Khởi tạo API
    api = VietnameseToneAPI()
    
    # Load model
    print("Đang load model...")
    model_loaded = api.load_model()
    
    if not model_loaded:
        print("\nKhông tìm thấy model. Bắt đầu training...")
        api.train_model(
            dict_path="data/Viet74K_clean.txt",
            corpus_path="data/cleaned_comments.txt",
            epochs=10,  # Giảm epochs để demo nhanh hơn
            batch_size=64
        )
    
    # Hiển thị thông tin model
    print("\n" + "="*65)
    print("THÔNG TIN MODEL")
    model_info = api.get_model_info()
    print(f"Model type: {model_info['model_type']}")
    print(f"Status: {'Loaded' if model_info['loaded'] else 'Not loaded'}")
    print("Features:")
    for feature in model_info['features']:
        print(f"  - {feature}")
    print("Strengths:")
    for strength in model_info['strengths']:
        print(f"  - {strength}")
    
    # Test cases
    test_cases = [
        "toi la sinh vien dai hoc",
        "ban co khoe khong",
        "hom nay troi rat dep",
        "cam on ban rat nhieu",
        "xin chao tat ca moi nguoi",
        "chung ta hay di an com"
    ]
    
    print("\n" + "="*65)
    print("DEMO KHÔI PHỤC DẤU")
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Input: '{text}'")
        
        start_time = time.time()
        results = api.restore_tones(text, max_results=3)
        elapsed_time = time.time() - start_time
        
        print("Results:")
        for j, (restored, score) in enumerate(results, 1):
            print(f"  {j}. {restored} (điểm: {score:.4f})")
        print(f"Thời gian: {elapsed_time*1000:.2f}ms")
    
    # Benchmark
    print("\n" + "="*65)
    print("BENCHMARK")
    benchmark_results = api.benchmark(test_cases[:4], iterations=3)
    
    # Demo interactive
    print("\n" + "="*65)
    print("CHẾ ĐỘ TƯƠNG TÁC")
    print("Nhập văn bản cần khôi phục dấu (hoặc 'quit' để thoát):")
    
    while True:
        try:
            text = input("\n> ").strip()
            if text.lower() in ['quit', 'q', 'exit']:
                break
            
            if not text:
                continue
            
            start_time = time.time()
            results = api.restore_tones(text, max_results=3)
            elapsed_time = time.time() - start_time
            
            print(f"\nKết quả:")
            for i, (restored, score) in enumerate(results, 1):
                print(f"  {i}. {restored} (điểm: {score:.4f})")
            print(f"Thời gian: {elapsed_time*1000:.2f}ms")
            
        except KeyboardInterrupt:
            break
    
    print("\nCảm ơn bạn đã sử dụng Vietnamese Neural Tone API!")

if __name__ == "__main__":
    main() 