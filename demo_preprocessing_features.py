import json
from data_preprocessing import VietnameseDataPreprocessor

class DemoPreprocessor:
    def __init__(self):
        self.preprocessor = VietnameseDataPreprocessor()
        # Load ngram dictionary
        print("Đang load từ điển n-gram...")
        self.preprocessor.load_ngram_dictionary()
        print(f"Đã load {len(self.preprocessor.ngram_dict)} entries\n")
    
    def demo_normalization(self):
        """Demo chuẩn hóa đầu vào"""
        print("=== DEMO CHUẨN HÓA ĐẦU VÀO ===")
        
        test_cases = [
            "CHAO BAN, TOI LA MINH!",
            "Hom Nay Troi Dep Qua.",
            "123 con GA, 456 con VIT.",
            "Email: test@example.com, Phone: 0123-456-789"
        ]
        
        for text in test_cases:
            normalized = self.preprocessor.normalize_text(text)
            case_info = self.preprocessor.preserve_case_info(text)
            restored = self.preprocessor.restore_case(normalized, case_info)
            
            print(f"Input:       {text}")
            print(f"Normalized:  {normalized}")
            print(f"Case info:   {case_info[:20]}{'...' if len(case_info) > 20 else ''}")
            print(f"Restored:    {restored}")
            print()
    
    def demo_d_character_handling(self):
        """Demo xử lý ký tự 'd' đặc biệt"""
        print("=== DEMO XỬ LÝ KÝ TỰ 'D' ĐẶC BIỆT ===")
        
        d_cases = [
            "dan toc",      # có thể là "dân tộc" 
            "doi song",     # có thể là "đời sống"
            "dam dong",     # có thể là "đám đông"
            "dung dung",    # có thể là "đứng dừng"
            "du lich",      # có thể là "du lịch"
            "dien thoai",   # có thể là "điện thoại"
        ]
        
        print("Các trường hợp 'd' có thể thành 'đ':")
        for text in d_cases:
            suggestions = self.preprocessor.process_text_with_ngrams(text)
            print(f"Input: '{text}'")
            for phrase, options in suggestions:
                if options:
                    print(f"  '{phrase}' -> {options[:5]}")
                else:
                    print(f"  '{phrase}' -> (không có gợi ý)")
            print()
    
    def demo_sliding_window_algorithm(self):
        """Demo thuật toán vùng trượt từ trái sang phải"""
        print("=== DEMO THUẬT TOÁN VÙNG TRƯỢT ===")
        
        test_sentences = [
            "cam on ban rat nhieu",
            "chuc mung nam moi",
            "anh hung khong dat dung vo",
            "toi yeu viet nam",
            "hoc tap roi se thanh cong"
        ]
        
        for sentence in test_sentences:
            print(f"Câu: '{sentence}'")
            print("Phân tích từng bước:")
            
            words = sentence.split()
            i = 0
            step = 1
            
            while i < len(words):
                print(f"  Bước {step}: Vị trí {i}")
                
                # Tìm n-gram dài nhất
                matched_key, suggestions, length = self.preprocessor.find_longest_ngram_match(words, i)
                
                if length > 0:
                    print(f"    Tìm thấy {length}-gram: '{matched_key}'")
                    print(f"    Gợi ý: {suggestions[:3]}{'...' if len(suggestions) > 3 else ''}")
                    i += length
                else:
                    word = words[i]
                    word_suggestions = self.preprocessor.get_word_suggestions(word)
                    print(f"    Từ đơn: '{word}'")
                    print(f"    Gợi ý: {word_suggestions[:3]}{'...' if len(word_suggestions) > 3 else ''}")
                    i += 1
                
                step += 1
            print()
    
    def demo_diacritic_penalty(self):
        """Demo giới hạn kết quả vô nghĩa (diacritic penalty)"""
        print("=== DEMO DIACRITIC PENALTY ===")
        
        # Tạo một số gợi ý giả có chứa ký tự không hợp lệ
        test_suggestions = [
            ["cảm ơn", "cám ơn"],  # hợp lệ
            ["tôi làm", "toi lam123", "tôi làm!!!"],  # có chứa số và ký tự lạ
            ["việt nam", "viet@nam", "việt#nam"],  # có ký tự đặc biệt lạ
            ["học tập", "học tập", "học†tập"],  # có ký tự unicode lạ
        ]
        
        print("Test lọc gợi ý không hợp lệ:")
        for i, suggestions in enumerate(test_suggestions):
            print(f"Test {i+1}: {suggestions}")
            filtered = self.preprocessor.filter_invalid_suggestions(suggestions)
            print(f"  Sau lọc: {filtered}")
            print()
    
    def demo_ngram_priority(self):
        """Demo ưu tiên n-gram dài hơn"""
        print("=== DEMO ƯU TIÊN N-GRAM DÀI ===")
        
        # Tìm một số case có cả ngram ngắn và dài
        test_cases = [
            "cam on ban",     # có "cam on" (2-gram) và có thể có "cam" (1-gram)
            "chuc mung nam moi",  # có nhiều n-gram chồng lấp
            "anh hung khong dat",  # test với thành ngữ
        ]
        
        for text in test_cases:
            print(f"Text: '{text}'")
            words = text.split()
            
            # Kiểm tra từng vị trí xem có n-gram nào khớp
            for i in range(len(words)):
                print(f"  Vị trí {i} ('{words[i]}'):")
                
                # Test các độ dài n-gram khác nhau
                for length in range(1, min(6, len(words) - i + 1)):
                    candidate = " ".join(words[i:i + length])
                    if candidate in self.preprocessor.ngram_dict:
                        suggestions = self.preprocessor.ngram_dict[candidate]
                        print(f"    {length}-gram '{candidate}': {suggestions[:2]}{'...' if len(suggestions) > 2 else ''}")
            print()
    
    def demo_context_vs_no_context(self):
        """Demo so sánh gợi ý có ngữ cảnh vs không ngữ cảnh"""
        print("=== DEMO NGỮ CẢNH VS KHÔNG NGỮ CẢNH ===")
        
        ambiguous_words = ["toi", "ban", "an", "di"]
        
        contexts = [
            "toi yeu viet nam",
            "ban be cua toi", 
            "an com nha",
            "di hoc ve"
        ]
        
        print("Gợi ý không có ngữ cảnh (từ đơn lẻ):")
        for word in ambiguous_words:
            suggestions = self.preprocessor.get_word_suggestions(word)
            print(f"  '{word}' -> {suggestions[:5]}{'...' if len(suggestions) > 5 else ''}")
        print()
        
        print("Gợi ý có ngữ cảnh (trong câu):")
        for context in contexts:
            print(f"Câu: '{context}'")
            analysis = self.preprocessor.process_text_with_ngrams(context)
            for phrase, options in analysis:
                if len(phrase.split()) == 1 and phrase in ambiguous_words:
                    print(f"  '{phrase}' (trong ngữ cảnh) -> {options[:3]}{'...' if len(options) > 3 else ''}")
        print()
    
    def run_all_demos(self):
        """Chạy tất cả demo"""
        print("VIETNAMESE ACCENT PREPROCESSING DEMO")
        print("=" * 50)
        print()
        
        self.demo_normalization()
        self.demo_d_character_handling()
        self.demo_sliding_window_algorithm()
        self.demo_diacritic_penalty()
        self.demo_ngram_priority()
        self.demo_context_vs_no_context()
        
        print("=" * 50)
        print("Hoàn thành tất cả demo!")

if __name__ == "__main__":
    demo = DemoPreprocessor()
    demo.run_all_demos() 