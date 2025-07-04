import json
import os
from pathlib import Path

def check_corpus_structure():
    """Kiểm tra cấu trúc file trong corpus_splitted"""
    print("=== KIỂM TRA CẤU TRÚC CORPUS_SPLITTED ===")
    
    corpus_dir = "corpus_splitted"
    if not os.path.exists(corpus_dir):
        print(f"Thư mục {corpus_dir} không tồn tại!")
        return
    
    files = [f for f in os.listdir(corpus_dir) if f.endswith('.json')]
    print(f"Tổng số file: {len(files)}")
    
    # Kiểm tra file đầu tiên
    if files:
        first_file = os.path.join(corpus_dir, files[0])
        try:
            with open(first_file, 'r', encoding='utf-8') as f:
                # Đọc từng dòng thay vì load toàn bộ file
                lines = []
                for i, line in enumerate(f):
                    lines.append(line.strip())
                    if i >= 10:  # Chỉ đọc 10 dòng đầu
                        break
                
                print(f"\nCấu trúc file {files[0]}:")
                print("10 dòng đầu:")
                for i, line in enumerate(lines):
                    print(f"Dòng {i+1}: {line[:100]}{'...' if len(line) > 100 else ''}")
                
                # Thử parse JSON
                f.seek(0)
                content = f.read(1000)  # Đọc 1000 ký tự đầu
                print(f"\n1000 ký tự đầu:")
                print(content)
                
        except Exception as e:
            print(f"Lỗi đọc file {first_file}: {e}")

def check_ngrams_structure():
    """Kiểm tra cấu trúc file trong ngrams"""
    print("\n=== KIỂM TRA CẤU TRÚC NGRAMS ===")
    
    ngrams_dir = "ngrams"
    if not os.path.exists(ngrams_dir):
        print(f"Thư mục {ngrams_dir} không tồn tại!")
        return
    
    files = sorted([f for f in os.listdir(ngrams_dir) if f.endswith('.json')])
    print(f"Tổng số file: {len(files)}")
    
    for filename in files[:3]:  # Kiểm tra 3 file đầu
        filepath = os.path.join(ngrams_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            print(f"\nFile {filename}:")
            print(f"  - Loại dữ liệu: {type(data)}")
            
            if isinstance(data, dict):
                print(f"  - Số lượng keys: {len(data)}")
                
                # Lấy 3 ví dụ đầu tiên
                items = list(data.items())[:3]
                print(f"  - 3 ví dụ đầu tiên:")
                for key, value in items:
                    print(f"    '{key}' -> {value}")
                    
        except Exception as e:
            print(f"Lỗi đọc file {filepath}: {e}")

def estimate_data_size():
    """Ước tính kích thước dữ liệu"""
    print("\n=== ƯỚC TÍNH KÍCH THƯỚC DỮ LIỆU ===")
    
    # Kiểm tra corpus
    corpus_dir = "corpus_splitted"
    if os.path.exists(corpus_dir):
        corpus_files = [f for f in os.listdir(corpus_dir) if f.endswith('.json')]
        total_size = 0
        for filename in corpus_files:
            filepath = os.path.join(corpus_dir, filename)
            size = os.path.getsize(filepath)
            total_size += size
        
        print(f"Corpus:")
        print(f"  - Số file: {len(corpus_files)}")
        print(f"  - Tổng kích thước: {total_size / (1024**3):.2f} GB")
        print(f"  - Kích thước trung bình: {total_size / len(corpus_files) / (1024**2):.2f} MB/file")
    
    # Kiểm tra ngrams
    ngrams_dir = "ngrams"
    if os.path.exists(ngrams_dir):
        ngrams_files = [f for f in os.listdir(ngrams_dir) if f.endswith('.json')]
        total_size = 0
        for filename in ngrams_files:
            filepath = os.path.join(ngrams_dir, filename)
            size = os.path.getsize(filepath)
            total_size += size
        
        print(f"N-grams:")
        print(f"  - Số file: {len(ngrams_files)}")
        print(f"  - Tổng kích thước: {total_size / (1024**2):.2f} MB")

def test_json_parsing():
    """Test parsing JSON với encoding khác nhau"""
    print("\n=== TEST PARSING JSON ===")
    
    # Test với file nhỏ nhất từ corpus
    corpus_dir = "corpus_splitted"
    files = os.listdir(corpus_dir)
    
    # Tìm file nhỏ nhất
    smallest_file = None
    smallest_size = float('inf')
    
    for filename in files:
        if filename.endswith('.json'):
            filepath = os.path.join(corpus_dir, filename)
            size = os.path.getsize(filepath)
            if size < smallest_size:
                smallest_size = size
                smallest_file = filepath
    
    if smallest_file:
        print(f"Testing với file nhỏ nhất: {smallest_file} ({smallest_size / (1024**2):.2f} MB)")
        
        # Test các encoding khác nhau
        encodings = ['utf-8', 'utf-8-sig', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(smallest_file, 'r', encoding=encoding) as f:
                    # Đọc vài dòng đầu
                    first_lines = []
                    for i, line in enumerate(f):
                        first_lines.append(line.strip())
                        if i >= 5:
                            break
                
                print(f"\nEncoding {encoding} - thành công:")
                for i, line in enumerate(first_lines):
                    print(f"  Dòng {i+1}: {line[:80]}{'...' if len(line) > 80 else ''}")
                
                # Thử parse JSON
                with open(smallest_file, 'r', encoding=encoding) as f:
                    sample = f.read(500)
                    print(f"  Sample 500 ký tự: {sample}")
                
                break
                
            except Exception as e:
                print(f"Encoding {encoding} - lỗi: {e}")

if __name__ == "__main__":
    check_corpus_structure()
    check_ngrams_structure()
    estimate_data_size()
    test_json_parsing() 