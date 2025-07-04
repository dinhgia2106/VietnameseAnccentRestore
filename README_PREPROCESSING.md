# Pipeline Tiền Xử Lý Dữ Liệu Tiếng Việt

## Tổng Quan

Pipeline này thực hiện tiền xử lý dữ liệu cho bài toán gán dấu thanh điệu tiếng Việt, bao gồm chuẩn hóa đầu vào, xây dựng từ điển n-gram và thuật toán gợi ý thông minh.

## Cấu Trúc Dữ Liệu

### 1. Corpus Data (`corpus_splitted/`)

- **Định dạng**: 233 file JSON (tổng ~38GB)
- **Cấu trúc**: Array of objects với format:

```json
{
  "input": "text khong dau",
  "target": "text có dấu"
}
```

- **Số lượng**: ~496,000 mẫu/file

### 2. N-grams Data (`ngrams/`)

- **Định dạng**: 17 file JSON (1-gram đến 17-gram)
- **Cấu trúc**: Dictionary mapping:

```json
{
  "khong dau": ["có dấu 1", "có dấu 2", ...]
}
```

- **Tổng số entries**: 104,763

## Tính Năng Chính

### 1. Chuẩn Hóa Đầu Vào

- **Lowercase transformation**: Giữ nguyên cấu trúc và vị trí
- **Case preservation**: Lưu thông tin chữ hoa để khôi phục
- **Character handling**: Bảo toàn dấu câu, số và ký tự đặc biệt

```python
# Ví dụ
input: "CHAO BAN, TOI LA MINH!"
normalized: "chao ban, toi la minh!"
case_info: [True, True, True, True, False, ...]
restored: "CHAO BAN, TOI LA MINH!"
```

### 2. Xử Lý Ký Tự 'D' Đặc Biệt

- Nhận diện các trường hợp 'd' có thể thành 'đ'
- Dựa vào từ điển và ngữ cảnh để đưa ra gợi ý

```python
# Ví dụ
"dan toc" -> "dân tộc"
"dien thoai" -> "điện thoại"
```

### 3. Thuật Toán Vùng Trượt (Sliding Window)

- **Ưu tiên n-gram dài**: Tìm cụm từ dài nhất trước
- **Xử lý từ trái sang phải**: Thuật toán greedy
- **Fallback**: Xử lý từ đơn nếu không tìm thấy n-gram

```python
# Ví dụ: "cam on ban rat nhieu"
# Bước 1: Tìm "cam on" (2-gram) -> ["cảm ơn", "cám ơn"]
# Bước 2: Tìm "ban" (1-gram) -> ["bạn", "bàn", "bán"]
# Bước 3: Tìm "rat" (1-gram) -> ["rất", "rắt"]
# Bước 4: Tìm "nhieu" (1-gram) -> ["nhiều", "nhiễu"]
```

### 4. Diacritic Penalty (Lọc Kết Quả Vô Nghĩa)

- Loại bỏ gợi ý chứa ký tự không hợp lệ
- Kiểm tra regex pattern cho tiếng Việt
- Ngăn chặn các kết hợp không tồn tại

```python
# Ví dụ
valid: ["cảm ơn", "cám ơn"]
invalid: ["viet@nam", "học†tập"] -> bị loại bỏ
```

### 5. Gợi Ý Có Ngữ Cảnh vs Không Ngữ Cảnh

- **Không ngữ cảnh**: Tất cả khả năng của từ đơn
- **Có ngữ cảnh**: Ưu tiên dựa trên n-gram surrounding

```python
# "toi" đơn lẻ -> ["tôi", "tới", "tội", "tỏi", ...]
# "toi" trong "cam on ban" -> ưu tiên theo n-gram
```

## Cách Sử Dụng

### 1. Chạy Pipeline Đầy Đủ

```bash
python data_preprocessing.py
```

### 2. Demo Các Tính Năng

```bash
python demo_preprocessing_features.py
```

### 3. Kiểm Tra Cấu Trúc Dữ Liệu

```bash
python check_data_structure.py
```

## Output

### Processed Data Format

```json
{
  "original_input": "Text gốc",
  "normalized_input": "text chuẩn hóa",
  "target": "Text có dấu",
  "target_normalized": "text có dấu chuẩn hóa",
  "case_info": [true, false, ...],
  "ngram_analysis": [("phrase", ["suggestions"]), ...],
  "input_chars": ["t", "e", "x", "t", ...],
  "target_chars": ["t", "ề", "x", "t", ...],
  "length": 123
}
```

### Files Được Tạo

- `train_processed.json`: Dữ liệu training (90%)
- `val_processed.json`: Dữ liệu validation (10%)

## Thống Kê Dữ Liệu

### Corpus Statistics

- **Train**: 18,000 mẫu
- **Validation**: 2,000 mẫu
- **Độ dài trung bình**: ~132 ký tự/câu
- **Range**: 10-4,135 ký tự

### N-gram Statistics

- **1-gram**: 2,800 entries (từ đơn)
- **2-gram**: 54,528 entries
- **3-gram**: 25,689 entries
- **...**: Giảm dần
- **17-gram**: 2 entries (thành ngữ dài)

## Ưu Điểm

1. **Xử lý quy mô lớn**: Streaming để tiết kiệm memory
2. **Thuật toán thông minh**: Ưu tiên n-gram dài, có ngữ cảnh
3. **Robust filtering**: Loại bỏ kết quả vô nghĩa
4. **Reproducible**: Seed cố định cho kết quả nhất quán
5. **Comprehensive**: Bao phủ từ 1-gram đến 17-gram

## Hạn Chế

1. **Ký tự 'd/đ'**: Chưa có logic phức tạp, dựa vào từ điển
2. **Memory usage**: Với dataset đầy đủ cần ~40GB RAM
3. **Context window**: Chỉ dựa vào n-gram, chưa có semantic context

## Hướng Phát Triển

1. **Deep learning integration**: Kết hợp với mô hình transformer
2. **Advanced d/đ handling**: Machine learning cho ký tự đặc biệt
3. **Dynamic weighting**: Trọng số dựa trên frequency và context
4. **Real-time processing**: Tối ưu cho inference nhanh

## Requirements

```bash
python >= 3.7
json (built-in)
collections (built-in)
typing (built-in)
re (built-in)
random (built-in)
os (built-in)
pathlib (built-in)
```

## Liên Hệ

Để hỗ trợ và phát triển thêm, vui lòng tham khảo documentation và code comments trong các file Python.
