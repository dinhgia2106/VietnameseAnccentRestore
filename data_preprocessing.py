import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import random
from pathlib import Path

class VietnameseDataPreprocessor:
    def __init__(self, corpus_dir: str = "corpus_splitted", ngrams_dir: str = "ngrams"):
        self.corpus_dir = corpus_dir
        self.ngrams_dir = ngrams_dir
        self.ngram_dict = {}
        self.word_suggestions = defaultdict(set)
        
    def normalize_text(self, text: str) -> str:
        """
        Chuẩn hóa văn bản: chuyển về lowercase, giữ nguyên khoảng trắng và ký tự đặc biệt
        """
        return text.lower()
    
    def handle_d_character(self, text: str) -> str:
        """
        Xử lý ký tự 'd' đặc biệt trong tiếng Việt
        Lưu ý: đây là trường hợp phức tạp cần ngữ cảnh để quyết định 'd' hay 'đ'
        """
        # Hiện tại giữ nguyên, sẽ để mô hình học từ dữ liệu
        return text
    
    def preserve_case_info(self, original_text: str) -> List[bool]:
        """
        Lưu thông tin về chữ hoa để khôi phục sau
        """
        return [c.isupper() for c in original_text]
    
    def restore_case(self, text: str, case_info: List[bool]) -> str:
        """
        Khôi phục chữ hoa dựa trên thông tin đã lưu
        """
        result = []
        for i, char in enumerate(text):
            if i < len(case_info) and case_info[i] and char.isalpha():
                result.append(char.upper())
            else:
                result.append(char)
        return ''.join(result)
    
    def load_corpus_data_streaming(self, max_samples: int = 50000) -> Tuple[List[Dict], List[Dict]]:
        """
        Đọc dữ liệu từ corpus_splitted với streaming để tiết kiệm memory
        """
        all_data = []
        
        # Lấy danh sách file và sắp xếp
        corpus_files = sorted([f for f in os.listdir(self.corpus_dir) if f.endswith('.json')])
        
        print(f"Tìm thấy {len(corpus_files)} file corpus")
        
        for filename in corpus_files:
            filepath = os.path.join(self.corpus_dir, filename)
            
            try:
                print(f"Đang đọc {filename}...")
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            if "input" in item and "target" in item:
                                all_data.append(item)
                                
                                # Giới hạn số lượng để tránh out of memory
                                if len(all_data) >= max_samples:
                                    print(f"Đã đạt giới hạn {max_samples} mẫu")
                                    break
                    
                    print(f"Đã đọc {filename}: +{len(data)} mẫu, tổng: {len(all_data)}")
                    
                    if len(all_data) >= max_samples:
                        break
                        
            except Exception as e:
                print(f"Lỗi đọc file {filename}: {e}")
                continue
        
        # Trộn ngẫu nhiên và chia 90/10
        random.seed(42)  # Để kết quả reproducible
        random.shuffle(all_data)
        split_idx = int(len(all_data) * 0.9)
        
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        print(f"Tổng dữ liệu: {len(all_data)}")
        print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
        
        return train_data, val_data
    
    def load_ngram_dictionary(self) -> Dict[str, List[str]]:
        """
        Đọc tất cả file n-gram và xây dựng từ điển gợi ý
        """
        ngram_dict = {}
        
        # Đọc từ 1-gram đến 17-gram (nếu tồn tại)
        for i in range(1, 18):
            ngram_file = os.path.join(self.ngrams_dir, f"{i}_gram.json")
            if os.path.exists(ngram_file):
                try:
                    print(f"Đang đọc {i}-gram...")
                    with open(ngram_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Cập nhật từ điển với n-gram mới
                    for key, suggestions in data.items():
                        if key not in ngram_dict:
                            ngram_dict[key] = []
                        
                        if isinstance(suggestions, list):
                            ngram_dict[key].extend(suggestions)
                        else:
                            ngram_dict[key].append(suggestions)
                    
                    print(f"Đã đọc {i}-gram: {len(data)} entries")
                    
                except Exception as e:
                    print(f"Lỗi đọc {ngram_file}: {e}")
        
        # Loại bỏ trùng lặp trong suggestions và sắp xếp theo độ dài
        for key in ngram_dict:
            unique_suggestions = list(set(ngram_dict[key]))
            # Sắp xếp theo độ dài giảm dần (ưu tiên gợi ý dài hơn)
            ngram_dict[key] = sorted(unique_suggestions, key=len, reverse=True)
            
        self.ngram_dict = ngram_dict
        print(f"Tổng từ điển n-gram: {len(ngram_dict)} entries")
        return ngram_dict
    
    def find_longest_ngram_match(self, words: List[str], start_pos: int) -> Tuple[str, List[str], int]:
        """
        Tìm n-gram dài nhất khớp với vị trí bắt đầu (thuật toán vùng trượt)
        """
        # Thử từ n-gram dài nhất xuống ngắn nhất
        max_ngram_length = min(17, len(words) - start_pos)
        
        for length in range(max_ngram_length, 0, -1):
            candidate = " ".join(words[start_pos:start_pos + length])
            
            if candidate in self.ngram_dict:
                return candidate, self.ngram_dict[candidate], length
        
        return "", [], 0
    
    def get_word_suggestions(self, word: str) -> List[str]:
        """
        Lấy gợi ý cho một từ đơn lẻ
        """
        if word in self.ngram_dict:
            return self.ngram_dict[word]
        return []
    
    def process_text_with_ngrams(self, text: str) -> List[Tuple[str, List[str]]]:
        """
        Xử lý văn bản bằng thuật toán vùng trượt từ trái sang phải
        """
        words = text.split()
        results = []
        i = 0
        
        while i < len(words):
            # Tìm n-gram dài nhất tại vị trí i
            matched_key, suggestions, length = self.find_longest_ngram_match(words, i)
            
            if length > 0:
                # Tìm thấy n-gram khớp
                results.append((matched_key, suggestions))
                i += length
            else:
                # Không tìm thấy n-gram, xử lý từ đơn lẻ
                word = words[i]
                word_suggestions = self.get_word_suggestions(word)
                results.append((word, word_suggestions))
                i += 1
        
        return results
    
    def filter_invalid_suggestions(self, suggestions: List[str]) -> List[str]:
        """
        Lọc bỏ các gợi ý không hợp lệ (diacritic penalty)
        """
        valid_suggestions = []
        
        for suggestion in suggestions:
            # Kiểm tra cơ bản: không chứa ký tự lạ
            if re.match(r'^[a-zA-ZÀ-ỹ\s,\.!\?;:]+$', suggestion):
                valid_suggestions.append(suggestion)
        
        return valid_suggestions
    
    def create_training_pairs(self, input_text: str, target_text: str) -> Dict:
        """
        Tạo cặp training từ input và target
        """
        # Chuẩn hóa input
        normalized_input = self.normalize_text(input_text)
        
        # Lưu thông tin case
        case_info = self.preserve_case_info(input_text)
        
        # Phân tích với n-grams
        ngram_analysis = self.process_text_with_ngrams(normalized_input)
        
        # Tạo character-level alignment (cho sequence labeling)
        input_chars = list(normalized_input)
        target_chars = list(target_text.lower())
        
        return {
            "original_input": input_text,
            "normalized_input": normalized_input,
            "target": target_text,
            "target_normalized": target_text.lower(),
            "case_info": case_info,
            "ngram_analysis": ngram_analysis,
            "input_chars": input_chars,
            "target_chars": target_chars,
            "length": len(input_chars)
        }
    
    def preprocess_dataset(self, 
                          max_samples: int = 50000,
                          save_train_path: str = "train_processed.json", 
                          save_val_path: str = "val_processed.json"):
        """
        Tiền xử lý toàn bộ dataset
        """
        print("Bắt đầu tiền xử lý dữ liệu...")
        
        # 1. Load n-gram dictionary
        print("1. Đang load từ điển n-gram...")
        self.load_ngram_dictionary()
        
        # 2. Load corpus data
        print("2. Đang load dữ liệu corpus...")
        train_data, val_data = self.load_corpus_data_streaming(max_samples)
        
        # 3. Process training data
        print("3. Đang xử lý dữ liệu training...")
        processed_train = self.process_dataset_batch(train_data)
        
        # 4. Process validation data
        print("4. Đang xử lý dữ liệu validation...")
        processed_val = self.process_dataset_batch(val_data)
        
        # 5. Save processed data
        print("5. Đang lưu dữ liệu đã xử lý...")
        with open(save_train_path, 'w', encoding='utf-8') as f:
            json.dump(processed_train, f, ensure_ascii=False, indent=2)
            
        with open(save_val_path, 'w', encoding='utf-8') as f:
            json.dump(processed_val, f, ensure_ascii=False, indent=2)
        
        print(f"Hoàn thành! Train: {len(processed_train)}, Val: {len(processed_val)}")
        
        # 6. Tạo thống kê
        self.generate_statistics(processed_train, processed_val)
        
        return processed_train, processed_val
    
    def process_dataset_batch(self, dataset: List[Dict]) -> List[Dict]:
        """
        Xử lý một batch dữ liệu
        """
        processed = []
        
        for i, item in enumerate(dataset):
            try:
                input_text = item["input"]
                target_text = item["target"]
                
                # Tạo training pair
                processed_item = self.create_training_pairs(input_text, target_text)
                processed.append(processed_item)
                
                if i % 5000 == 0 and i > 0:
                    print(f"Đã xử lý {i}/{len(dataset)} mẫu")
                    
            except Exception as e:
                print(f"Lỗi xử lý mẫu {i}: {e}")
                continue
        
        return processed
    
    def generate_statistics(self, train_data: List[Dict], val_data: List[Dict]):
        """
        Tạo thống kê về dữ liệu
        """
        print("\n=== THỐNG KÊ DỮ LIỆU ===")
        
        # Thống kê độ dài câu
        train_lengths = [item["length"] for item in train_data]
        val_lengths = [item["length"] for item in val_data]
        
        print(f"Độ dài câu trung bình:")
        print(f"  Train: {sum(train_lengths)/len(train_lengths):.1f} ký tự")
        print(f"  Val: {sum(val_lengths)/len(val_lengths):.1f} ký tự")
        
        print(f"Độ dài câu min/max:")
        print(f"  Train: {min(train_lengths)}/{max(train_lengths)}")
        print(f"  Val: {min(val_lengths)}/{max(val_lengths)}")
        
        # Thống kê từ điển
        print(f"\nThống kê từ điển n-gram:")
        print(f"  Tổng số entries: {len(self.ngram_dict)}")
        
        # Đếm theo độ dài n-gram
        ngram_counts = defaultdict(int)
        for key in self.ngram_dict.keys():
            ngram_length = len(key.split())
            ngram_counts[ngram_length] += 1
        
        for length in sorted(ngram_counts.keys()):
            print(f"  {length}-gram: {ngram_counts[length]} entries")

def main():
    # Khởi tạo preprocessor
    preprocessor = VietnameseDataPreprocessor()
    
    # Chạy tiền xử lý với giới hạn 20000 mẫu để test
    train_data, val_data = preprocessor.preprocess_dataset(max_samples=20000)
    
    # Ví dụ kiểm tra kết quả
    if train_data:
        print("\n=== VÍ DỤ MẪU ĐÃ XỬ LÝ ===")
        sample = train_data[0]
        print(f"Input gốc: {sample['original_input']}")
        print(f"Input chuẩn hóa: {sample['normalized_input']}")
        print(f"Target: {sample['target']}")
        print(f"Độ dài: {sample['length']} ký tự")
        print(f"Phân tích n-gram (3 đầu tiên): {sample['ngram_analysis'][:3]}")
        
        # Test thuật toán gợi ý
        print(f"\n=== TEST THUẬT TOÁN GỢI Ý ===")
        test_input = "cam on ban rat nhieu"
        print(f"Input test: {test_input}")
        suggestions = preprocessor.process_text_with_ngrams(test_input)
        print(f"Gợi ý:")
        for phrase, options in suggestions:
            print(f"  '{phrase}' -> {options[:3] if len(options) > 3 else options}")

if __name__ == "__main__":
    main() 