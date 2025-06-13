# Mô hình Phục hồi Dấu Tiếng Việt - A-TCN

Mô hình A-TCN (Acausal Temporal Convolutional Network) để phục hồi dấu tiếng Việt từ văn bản không dấu. Mô hình được tối ưu cho tốc độ cao và có thể chạy trên thiết bị giới hạn tài nguyên.

## Dữ liệu

Tải về ở [đây](https://drive.google.com/file/d/1ovLbpvzSGrS4NDxZu8Ftdgc73uHzNQJf/view) và đây [đây](https://github.com/duyet/vietnamese-wordlist/blob/master/Viet74K.txt)

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
├── data/
│   ├── corpus-full.txt          # Corpus đầy đủ
│   ├── create_sample.py         # Script tạo sample
│   └── training_pairs.txt       # Dữ liệu huấn luyện (sau khi xử lý)
├── model_architecture.py       # Định nghĩa mô hình A-TCN
├── data_preprocessing.py       # Xử lý dữ liệu
├── train_model.py             # Script huấn luyện
├── demo.py                    # Demo và đánh giá
├── requirements.txt           # Dependencies
└── models/                   # Thư mục lưu mô hình
    ├── best_model.pth        # Mô hình tốt nhất
    ├── accent_restorer.onnx  # Mô hình ONNX
    └── training_history.json # Lịch sử huấn luyện
```

## Sử dụng

### 1. Xử lý dữ liệu

```bash
python data_preprocessing.py
```

### 2. Huấn luyện mô hình

```bash
python train_model.py
```

### 3. Demo và đánh giá

```bash
python demo.py
```

### 4. Sử dụng trong code

```python
from model_architecture import VietnameseAccentRestorer

# Tải mô hình
restorer = VietnameseAccentRestorer("models/best_model.pth")

# Phục hồi dấu
text_no_accent = "toi di hoc"
text_with_accent = restorer.predict(text_no_accent)
print(text_with_accent)  # "tôi đi học"
```

## Kiến trúc mô hình

Mô hình A-TCN bao gồm:

- **Embedding Layer**: Chuyển ký tự thành vector
- **TCN Blocks**: 4 lớp với dilated convolution (dilation: 1, 2, 4, 8)
- **Residual Connections**: Giúp huấn luyện ổn định
- **Classification Layer**: Phân loại cho từng ký tự

### Ưu điểm của A-TCN

1. **Tốc độ**: Convolution song song thay vì RNN tuần tự
2. **Ngữ cảnh**: Dilated convolution cho phạm vi lớn
3. **Nhẹ**: Ít tham số hơn Transformer
4. **Acausal**: Xem cả ngữ cảnh trước và sau

## Triển khai

### Web (ONNX Runtime Web)

```javascript
// Load ONNX model
const session = await ort.InferenceSession.create("accent_restorer.onnx");

// Inference
const results = await session.run({
  input: new ort.Tensor("int64", inputIds, [1, inputIds.length]),
});
```

### Mobile (TensorFlow Lite)

```kotlin
// Android
val interpreter = Interpreter(loadModelFile())
interpreter.run(inputArray, outputArray)
```

### API Server

```python
from flask import Flask, request, jsonify
from model_architecture import VietnameseAccentRestorer

app = Flask(__name__)
restorer = VietnameseAccentRestorer("models/best_model.pth")

@app.route('/restore', methods=['POST'])
def restore_accent():
    text = request.json['text']
    result = restorer.predict(text)
    return jsonify({'result': result})
```

## Benchmark

### Hiệu suất dự kiến

- **Độ chính xác**: 95-99% ký tự
- **Tốc độ**: >1000 ký tự/giây (CPU)
- **Kích thước**: ~10MB (ONNX)
- **Latency**: <50ms cho câu 50 ký tự

### So sánh với các phương pháp khác

| Phương pháp | Độ chính xác | Tốc độ    | Kích thước | Triển khai |
| ----------- | ------------ | --------- | ---------- | ---------- |
| A-TCN       | 95-99%       | Rất nhanh | ~10MB      | Dễ         |
| BiLSTM      | 95-98%       | Chậm      | ~15MB      | Trung bình |
| DistilBERT  | 98-99%       | Chậm      | ~170MB     | Khó        |
| XLM-RoBERTa | 99%+         | Rất chậm  | ~1GB       | Rất khó    |

## Cải tiến có thể

1. **Attention Mechanism**: Thêm self-attention
2. **Multi-task Learning**: Kết hợp với word segmentation
3. **Knowledge Distillation**: Từ mô hình BERT lớn
4. **Data Augmentation**: Tăng cường dữ liệu
5. **Ensemble**: Kết hợp nhiều mô hình

## Tài liệu tham khảo

- Alqahtani et al. (EMNLP 2019): "Efficient CNN for Diacritics Restoration"
- Csanády & Lukács (2022): "Acausal TCN for Diacritics"
- Bai et al. (2018): "An Empirical Evaluation of Generic Convolutional and Recurrent Networks"

## License

MIT License
