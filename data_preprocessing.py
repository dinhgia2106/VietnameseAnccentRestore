import re
import unicodedata
from typing import List, Tuple
import random
from tqdm import tqdm

class VietnameseDataPreprocessor:
    """
    Xử lý dữ liệu tiếng Việt để tạo tập huấn luyện cho mô hình phục hồi dấu
    """
    
    def __init__(self):
        # Bản đồ chuyển đổi từ có dấu sang không dấu
        self.accent_map = {
            'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ắ': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ấ': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'í': 'i', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ó': 'o', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ố': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ớ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ú': 'u', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ứ': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ý': 'y', 'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd',
            'Á': 'A', 'À': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
            'Ă': 'A', 'Ắ': 'A', 'Ằ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
            'Â': 'A', 'Ấ': 'A', 'Ầ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
            'É': 'E', 'È': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
            'Ê': 'E', 'Ế': 'E', 'Ề': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
            'Í': 'I', 'Ì': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
            'Ó': 'O', 'Ò': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
            'Ô': 'O', 'Ố': 'O', 'Ồ': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
            'Ơ': 'O', 'Ớ': 'O', 'Ờ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
            'Ú': 'U', 'Ù': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
            'Ư': 'U', 'Ứ': 'U', 'Ừ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
            'Ý': 'Y', 'Ỳ': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
            'Đ': 'D'
        }
    
    def remove_accents(self, text: str) -> str:
        """
        Loại bỏ dấu trong văn bản tiếng Việt
        """
        result = ""
        for char in text:
            if char in self.accent_map:
                result += self.accent_map[char]
            else:
                result += char
        return result
    
    def clean_text(self, text: str) -> str:
        """
        Làm sạch văn bản
        """
        # Chuẩn hóa unicode
        text = unicodedata.normalize('NFC', text)
        
        # Loại bỏ ký tự đặc biệt không cần thiết
        text = re.sub(r'[^\w\s.,!?;:()-\[\]{}"\'/]', '', text)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def create_training_pairs(self, sentences: List[str]) -> List[Tuple[str, str]]:
        """
        Tạo cặp (không dấu, có dấu) để huấn luyện
        """
        pairs = []
        
        for sentence in tqdm(sentences, desc="Tạo training pairs"):
            # Làm sạch
            clean_sentence = self.clean_text(sentence)
            
            if len(clean_sentence) < 5 or len(clean_sentence) > 500:
                continue
                
            # Tạo phiên bản không dấu
            no_accent = self.remove_accents(clean_sentence)
            
            # Chỉ giữ lại nếu có sự khác biệt (có ít nhất 1 ký tự có dấu)
            if no_accent != clean_sentence:
                pairs.append((no_accent, clean_sentence))
        
        return pairs
    
    def split_by_character(self, pairs: List[Tuple[str, str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Chia văn bản thành từng ký tự để training character-level
        """
        input_sequences = []
        target_sequences = []
        
        for no_accent, with_accent in pairs:
            if len(no_accent) == len(with_accent):
                input_sequences.append(list(no_accent))
                target_sequences.append(list(with_accent))
        
        return input_sequences, target_sequences
    
    def augment_data(self, pairs: List[Tuple[str, str]], augment_ratio: float = 0.3) -> List[Tuple[str, str]]:
        """
        Tăng cường dữ liệu bằng cách thêm noise
        """
        augmented_pairs = pairs.copy()
        
        for no_accent, with_accent in pairs:
            if random.random() < augment_ratio:
                # Thêm một ít noise bằng cách thay đổi ngẫu nhiên một số ký tự
                noisy_input = self.add_character_noise(no_accent, noise_ratio=0.05)
                augmented_pairs.append((noisy_input, with_accent))
        
        return augmented_pairs
    
    def add_character_noise(self, text: str, noise_ratio: float = 0.05) -> str:
        """
        Thêm noise vào text bằng cách thay đổi ngẫu nhiên một số ký tự
        """
        chars = list(text)
        num_chars_to_change = max(1, int(len(chars) * noise_ratio))
        
        for _ in range(num_chars_to_change):
            if len(chars) > 0:
                idx = random.randint(0, len(chars) - 1)
                if chars[idx].isalpha():
                    # Thay bằng ký tự ngẫu nhiên cùng case
                    if chars[idx].islower():
                        chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
                    else:
                        chars[idx] = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        return ''.join(chars)
    
    def process_corpus(self, corpus_file: str, output_file: str, max_samples: int = None):
        """
        Xử lý corpus và lưu thành tập huấn luyện
        """
        print(f"Đọc corpus từ: {corpus_file}")
        
        sentences = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                line = line.strip()
                if line:
                    sentences.append(line)
        
        print(f"Đã đọc {len(sentences)} câu")
        
        # Tạo training pairs
        pairs = self.create_training_pairs(sentences)
        print(f"Tạo được {len(pairs)} training pairs")
        
        # Tăng cường dữ liệu
        pairs = self.augment_data(pairs)
        print(f"Sau khi augment: {len(pairs)} training pairs")
        
        # Lưu kết quả
        with open(output_file, 'w', encoding='utf-8') as f:
            for no_accent, with_accent in pairs:
                f.write(f"{no_accent}\t{with_accent}\n")
        
        print(f"Đã lưu training data tại: {output_file}")
        
        return pairs

def main():
    """
    Hàm chính để xử lý dữ liệu
    """
    preprocessor = VietnameseDataPreprocessor()
    
    # Xử lý corpus sample
    corpus_file = "data/corpus-sample-100k.txt"
    output_file = "data/training_pairs.txt"
    
    pairs = preprocessor.process_corpus(corpus_file, output_file, max_samples=100000)
    
    # Thống kê
    total_chars = sum(len(pair[0]) for pair in pairs)
    avg_length = total_chars / len(pairs) if pairs else 0
    
    print(f"\nThống kê:")
    print(f"- Số lượng training pairs: {len(pairs)}")
    print(f"- Độ dài trung bình: {avg_length:.1f} ký tự")
    print(f"- Tổng số ký tự: {total_chars:,}")
    
    # Ví dụ
    print(f"\nVí dụ training pairs:")
    for i in range(min(5, len(pairs))):
        no_accent, with_accent = pairs[i]
        print(f"Input:  {no_accent}")
        print(f"Target: {with_accent}")
        print()

if __name__ == "__main__":
    main() 