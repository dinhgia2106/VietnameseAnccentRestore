import re
import unicodedata
from typing import List, Tuple
import random
from tqdm import tqdm
import json
import os

class EnhancedVietnameseDataPreprocessor:
    """
    Enhanced preprocessor với advanced techniques cho dữ liệu tiếng Việt
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
        
        # Từ vựng phổ biến để kiểm tra quality
        self.common_vietnamese_words = {
            'tôi', 'bạn', 'anh', 'chị', 'em', 'ông', 'bà', 'con',
            'người', 'ngày', 'năm', 'tháng', 'tuần', 'giờ', 'phút',
            'việt', 'nam', 'hà', 'nội', 'sài', 'gòn', 'thành', 'phố',
            'nhà', 'trường', 'công', 'ty', 'làm', 'việc', 'học', 'tập',
            'đi', 'đến', 'về', 'ra', 'vào', 'lên', 'xuống', 'qua',
            'rất', 'nhiều', 'ít', 'lớn', 'nhỏ', 'tốt', 'xấu', 'đẹp'
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
        Làm sạch văn bản với enhanced cleaning
        """
        # Chuẩn hóa unicode
        text = unicodedata.normalize('NFC', text)
        
        # Loại bỏ URL, email
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Loại bỏ số điện thoại
        text = re.sub(r'\b\d{10,11}\b', '', text)
        
        # Loại bỏ ký tự đặc biệt không cần thiết nhưng giữ dấu câu cơ bản
        text = re.sub(r'[^\w\s.,!?;:()"\'-]', '', text)
        
        # Chuẩn hóa dấu câu
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def is_good_quality_sentence(self, sentence: str) -> bool:
        """
        Kiểm tra chất lượng câu
        """
        # Độ dài phù hợp
        if len(sentence) < 10 or len(sentence) > 300:
            return False
        
        # Có ít nhất một từ tiếng Việt phổ biến
        words = sentence.lower().split()
        has_vietnamese_word = any(self.remove_accents(word) in 
                                {'toi', 'ban', 'anh', 'chi', 'em', 'viet', 'nam'} 
                                for word in words)
        
        # Tỷ lệ ký tự có dấu hợp lý (5-40%)
        total_chars = len(sentence.replace(' ', ''))
        accented_chars = sum(1 for char in sentence if char in self.accent_map)
        if total_chars > 0:
            accent_ratio = accented_chars / total_chars
            if accent_ratio < 0.05 or accent_ratio > 0.4:
                return False
        
        # Không có quá nhiều số
        digit_ratio = sum(1 for char in sentence if char.isdigit()) / len(sentence)
        if digit_ratio > 0.2:
            return False
        
        return has_vietnamese_word
    
    def create_training_pairs(self, sentences: List[str]) -> List[Tuple[str, str]]:
        """
        Tạo cặp (không dấu, có dấu) với quality filtering
        """
        pairs = []
        
        for sentence in tqdm(sentences, desc="Tạo training pairs"):
            # Làm sạch
            clean_sentence = self.clean_text(sentence)
            
            # Kiểm tra chất lượng
            if not self.is_good_quality_sentence(clean_sentence):
                continue
                
            # Tạo phiên bản không dấu
            no_accent = self.remove_accents(clean_sentence)
            
            # Chỉ giữ lại nếu có sự khác biệt
            if no_accent != clean_sentence:
                pairs.append((no_accent, clean_sentence))
        
        return pairs
    
    def advanced_augmentation(self, pairs: List[Tuple[str, str]], augment_ratio: float = 0.5) -> List[Tuple[str, str]]:
        """
        Advanced data augmentation với nhiều techniques
        """
        augmented_pairs = pairs.copy()
        
        for no_accent, with_accent in tqdm(pairs, desc="Augmenting data"):
            if random.random() < augment_ratio:
                # 1. Character-level noise
                if random.random() < 0.3:
                    noisy_input = self.add_character_noise(no_accent, noise_ratio=0.03)
                    augmented_pairs.append((noisy_input, with_accent))
                
                # 2. Case variation
                if random.random() < 0.2:
                    case_varied = self.vary_case(no_accent)
                    if case_varied != no_accent:
                        augmented_pairs.append((case_varied, with_accent))
                
                # 3. Punctuation variation
                if random.random() < 0.2:
                    punct_varied = self.vary_punctuation(no_accent)
                    if punct_varied != no_accent:
                        augmented_pairs.append((punct_varied, with_accent))
                
                # 4. Partial accent restoration (simulation of partial errors)
                if random.random() < 0.1:
                    partial_accent = self.create_partial_accent(no_accent, with_accent)
                    if partial_accent != no_accent and partial_accent != with_accent:
                        augmented_pairs.append((partial_accent, with_accent))
        
        return augmented_pairs
    
    def add_character_noise(self, text: str, noise_ratio: float = 0.03) -> str:
        """
        Thêm noise ký tự một cách thông minh
        """
        chars = list(text)
        num_chars_to_change = max(1, int(len(chars) * noise_ratio))
        
        # Common typing errors in Vietnamese
        vietnamese_confusion = {
            'a': ['ă', 'â'], 'ă': ['a', 'â'], 'â': ['a', 'ă'],
            'e': ['ê'], 'ê': ['e'],
            'o': ['ô', 'ơ'], 'ô': ['o', 'ơ'], 'ơ': ['o', 'ô'],
            'u': ['ư'], 'ư': ['u'],
            'd': ['đ'], 'đ': ['d']
        }
        
        for _ in range(num_chars_to_change):
            if len(chars) > 0:
                idx = random.randint(0, len(chars) - 1)
                char = chars[idx].lower()
                
                if char in vietnamese_confusion:
                    # Vietnamese-specific confusion
                    chars[idx] = random.choice(vietnamese_confusion[char])
                elif char.isalpha():
                    # General character confusion
                    if random.random() < 0.5:
                        # Adjacent key error
                        adjacent_keys = {
                            'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf',
                            't': 'ryg', 'y': 'tuh', 'u': 'yij', 'i': 'uok',
                            'o': 'ipl', 'p': 'ol'
                        }
                        if char in adjacent_keys:
                            chars[idx] = random.choice(adjacent_keys[char])
        
        return ''.join(chars)
    
    def vary_case(self, text: str) -> str:
        """
        Thay đổi case của text
        """
        if random.random() < 0.5:
            return text.title()  # Title case
        else:
            return text.upper()   # Upper case
    
    def vary_punctuation(self, text: str) -> str:
        """
        Thay đổi dấu câu
        """
        # Thêm hoặc bỏ dấu câu cuối
        if text and text[-1] not in '.!?':
            if random.random() < 0.5:
                text += random.choice('.!?')
        elif text and text[-1] in '.!?':
            if random.random() < 0.3:
                text = text[:-1]
        
        return text
    
    def create_partial_accent(self, no_accent: str, with_accent: str) -> str:
        """
        Tạo partial accent restoration để simulate lỗi model
        """
        if len(no_accent) != len(with_accent):
            return no_accent
        
        result = list(no_accent)
        # Randomly restore 30-70% of accents
        restore_ratio = random.uniform(0.3, 0.7)
        
        for i, (no_acc_char, acc_char) in enumerate(zip(no_accent, with_accent)):
            if no_acc_char != acc_char and random.random() < restore_ratio:
                result[i] = acc_char
        
        return ''.join(result)
    
    def create_balanced_dataset(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Tạo balanced dataset theo độ dài câu
        """
        # Phân loại theo độ dài
        short_pairs = [p for p in pairs if len(p[0]) <= 50]
        medium_pairs = [p for p in pairs if 50 < len(p[0]) <= 150]
        long_pairs = [p for p in pairs if len(p[0]) > 150]
        
        print(f"Short sentences (≤50): {len(short_pairs)}")
        print(f"Medium sentences (50-150): {len(medium_pairs)}")
        print(f"Long sentences (>150): {len(long_pairs)}")
        
        # Balance dataset
        min_count = min(len(short_pairs), len(medium_pairs), len(long_pairs))
        if min_count > 0:
            balanced_pairs = (
                random.sample(short_pairs, min(len(short_pairs), min_count * 2)) +
                random.sample(medium_pairs, min(len(medium_pairs), min_count * 2)) +
                random.sample(long_pairs, min(len(long_pairs), min_count))
            )
        else:
            balanced_pairs = pairs
        
        random.shuffle(balanced_pairs)
        return balanced_pairs
    
    def process_corpus(self, corpus_file: str, output_file: str, max_samples: int = None):
        """
        Enhanced corpus processing
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
        
        # Tạo training pairs với quality filtering
        pairs = self.create_training_pairs(sentences)
        print(f"Tạo được {len(pairs)} quality training pairs")
        
        # Advanced augmentation
        pairs = self.advanced_augmentation(pairs, augment_ratio=0.4)
        print(f"Sau khi augment: {len(pairs)} training pairs")
        
        # Balance dataset
        pairs = self.create_balanced_dataset(pairs)
        print(f"Sau khi balance: {len(pairs)} training pairs")
        
        # Lưu kết quả
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for no_accent, with_accent in pairs:
                f.write(f"{no_accent}\t{with_accent}\n")
        
        # Save metadata
        metadata = {
            'total_pairs': len(pairs),
            'avg_length': sum(len(pair[0]) for pair in pairs) / len(pairs),
            'quality_filtered': True,
            'augmentation_ratio': 0.4,
            'balanced': True
        }
        
        metadata_file = output_file.replace('.txt', '_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Đã lưu enhanced training data tại: {output_file}")
        print(f"Metadata lưu tại: {metadata_file}")
        
        return pairs

def main():
    """
    Hàm chính để xử lý dữ liệu enhanced
    """
    preprocessor = EnhancedVietnameseDataPreprocessor()
    
    # Xử lý corpus với enhanced techniques
    corpus_file = "data/corpus-full.txt"
    output_file = "data/enhanced_training_pairs.txt"
    
    pairs = preprocessor.process_corpus(corpus_file, output_file, max_samples=100000)
    
    # Thống kê chi tiết
    total_chars = sum(len(pair[0]) for pair in pairs)
    avg_length = total_chars / len(pairs) if pairs else 0
    
    # Phân tích accent distribution
    accent_counts = {}
    for _, with_accent in pairs:
        for char in with_accent:
            if char in preprocessor.accent_map:
                accent_counts[char] = accent_counts.get(char, 0) + 1
    
    print(f"\nThống kê enhanced dataset:")
    print(f"- Số lượng training pairs: {len(pairs):,}")
    print(f"- Độ dài trung bình: {avg_length:.1f} ký tự")
    print(f"- Tổng số ký tự: {total_chars:,}")
    print(f"- Số loại ký tự có dấu: {len(accent_counts)}")
    
    # Top accented characters
    top_accents = sorted(accent_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 ký tự có dấu phổ biến:")
    for char, count in top_accents:
        print(f"  {char}: {count:,}")
    
    # Ví dụ
    print(f"\nVí dụ enhanced training pairs:")
    for i in range(min(3, len(pairs))):
        no_accent, with_accent = pairs[i]
        print(f"Input:  {no_accent}")
        print(f"Target: {with_accent}")
        print()

if __name__ == "__main__":
    main() 