import torch
import time
import os
from model_architecture import VietnameseAccentRestorer

class VietnameseAccentDemo:
    """
    Demo cho mô hình phục hồi dấu tiếng Việt
    """
    
    def __init__(self, model_path=None):
        """
        Khởi tạo demo
        """
        print("Đang tải mô hình...")
        self.restorer = VietnameseAccentRestorer(model_path)
        
        if model_path and os.path.exists(model_path):
            print(f"Đã tải mô hình từ: {model_path}")
        else:
            print("Sử dụng mô hình chưa được huấn luyện (chỉ để test kiến trúc)")
        
        # Các câu test mẫu
        self.test_sentences = [
            "toi di hoc",
            "chung ta se thanh cong",
            "viet nam la dat nuoc xinh dep",
            "hom nay troi dep",
            "cam on ban rat nhieu",
            "xin chao moi nguoi",
            "chuc ban ngay tot lanh",
            "toi thich hoc tieng viet",
            "ha noi la thu do cua viet nam",
            "pho la mon an truyen thong"
        ]
        
        # Kết quả mong đợi (để so sánh)
        self.expected_results = [
            "tôi đi học",
            "chúng ta sẽ thành công",
            "việt nam là đất nước xinh đẹp",
            "hôm nay trời đẹp",
            "cảm ơn bạn rất nhiều",
            "xin chào mọi người",
            "chúc bạn ngày tốt lành",
            "tôi thích học tiếng việt",
            "hà nội là thủ đô của việt nam",
            "phở là món ăn truyền thống"
        ]
    
    def predict_single(self, text, verbose=True):
        """
        Dự đoán cho một câu
        """
        start_time = time.time()
        
        try:
            result = self.restorer.predict(text)
            inference_time = time.time() - start_time
            
            if verbose:
                print(f"Input:  {text}")
                print(f"Output: {result}")
                print(f"Thời gian: {inference_time*1000:.2f}ms")
                print("-" * 50)
            
            return result, inference_time
            
        except Exception as e:
            print(f"Lỗi khi xử lý: {text}")
            print(f"Chi tiết lỗi: {e}")
            return text, 0
    
    def run_batch_test(self):
        """
        Chạy test trên nhiều câu
        """
        print("DEMO PHỤC HỒI DẤU TIẾNG VIỆT")
        print("=" * 80)
        print(f"Mô hình A-TCN - Vocabulary size: {self.restorer.vocab_size}")
        print(f"Số tham số: {sum(p.numel() for p in self.restorer.model.parameters()):,}")
        print("=" * 80)
        
        total_time = 0
        total_chars = 0
        correct_predictions = 0
        
        for i, text in enumerate(self.test_sentences):
            print(f"\nTest {i+1}:")
            result, inference_time = self.predict_single(text)
            
            total_time += inference_time
            total_chars += len(text)
            
            # So sánh với kết quả mong đợi (nếu có)
            if i < len(self.expected_results):
                expected = self.expected_results[i]
                if result == expected:
                    correct_predictions += 1
                    print(f"Kết quả: ĐÚNG")
                else:
                    print(f"Kết quả: SAI")
                    print(f"Mong đợi: {expected}")
        
        # Thống kê
        print("\n" + "=" * 80)
        print("THỐNG KÊ HIỆU SUẤT")
        print("=" * 80)
        print(f"Tổng thời gian: {total_time*1000:.2f}ms")
        print(f"Thời gian trung bình: {total_time*1000/len(self.test_sentences):.2f}ms/câu")
        print(f"Tốc độ xử lý: {total_chars/total_time:.0f} ký tự/giây")
        
        if self.expected_results:
            accuracy = correct_predictions / min(len(self.test_sentences), len(self.expected_results))
            print(f"Độ chính xác: {accuracy*100:.1f}% ({correct_predictions}/{min(len(self.test_sentences), len(self.expected_results))})")
    
    def interactive_mode(self):
        """
        Chế độ tương tác
        """
        print("\n" + "=" * 80)
        print("CHẾ ĐỘ TƯƠNG TÁC")
        print("=" * 80)
        print("Nhập văn bản không dấu để phục hồi dấu.")
        print("Gõ 'quit' hoặc 'exit' để thoát.")
        print("-" * 80)
        
        while True:
            try:
                text = input("\nNhập văn bản: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Tạm biệt!")
                    break
                
                if not text:
                    continue
                
                result, inference_time = self.predict_single(text, verbose=False)
                print(f"Kết quả: {result}")
                print(f"Thời gian: {inference_time*1000:.2f}ms")
                
            except KeyboardInterrupt:
                print("\nTạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi: {e}")
    
    def benchmark_speed(self, num_iterations=100):
        """
        Benchmark tốc độ xử lý
        """
        print("\n" + "=" * 80)
        print("BENCHMARK TỐC ĐỘ")
        print("=" * 80)
        
        test_text = "toi di hoc va lam viec tai ha noi"
        print(f"Text test: {test_text}")
        print(f"Số lần lặp: {num_iterations}")
        print("-" * 80)
        
        # Warm up
        for _ in range(5):
            self.restorer.predict(test_text)
        
        # Benchmark
        start_time = time.time()
        for i in range(num_iterations):
            if i % 10 == 0:
                print(f"Progress: {i}/{num_iterations}", end="\r")
            self.restorer.predict(test_text)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        chars_per_sec = len(test_text) * num_iterations / total_time
        
        print(f"\nKết quả benchmark:")
        print(f"- Tổng thời gian: {total_time:.2f}s")
        print(f"- Thời gian trung bình: {avg_time*1000:.2f}ms/câu")
        print(f"- Tốc độ xử lý: {chars_per_sec:.0f} ký tự/giây")
        print(f"- Throughput: {num_iterations/total_time:.1f} câu/giây")

def main():
    """
    Hàm chính
    """
    # Kiểm tra model đã được huấn luyện
    model_path = "models/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Cảnh báo: Không tìm thấy model tại {model_path}")
        print("Sử dụng mô hình chưa huấn luyện để demo kiến trúc.")
        model_path = None
    
    # Khởi tạo demo
    demo = VietnameseAccentDemo(model_path)
    
    while True:
        print("\n" + "=" * 80)
        print("DEMO PHỤC HỒI DẤU TIẾNG VIỆT")
        print("=" * 80)
        print("Chọn chức năng:")
        print("1. Chạy test batch")
        print("2. Chế độ tương tác")
        print("3. Benchmark tốc độ")
        print("4. Thoát")
        print("-" * 80)
        
        try:
            choice = input("Nhập lựa chọn (1-4): ").strip()
            
            if choice == '1':
                demo.run_batch_test()
            elif choice == '2':
                demo.interactive_mode()
            elif choice == '3':
                demo.benchmark_speed()
            elif choice == '4':
                print("Tạm biệt!")
                break
            else:
                print("Lựa chọn không hợp lệ. Vui lòng chọn 1-4.")
                
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main() 