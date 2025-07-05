# Vietnamese Accent Restoration using A-TCN

Dự án khôi phục dấu thanh tiếng Việt sử dụng mô hình A-TCN (Acausal Temporal Convolutional Network).

## Cấu Trúc Dự Án

```
vietnamese_accent_restoration/
├── src/                          # Thư mục source code chính
│   ├── models/                   # Mô hình A-TCN
│   │   ├── __init__.py
│   │   ├── atcn.py              # Kiến trúc A-TCN
│   │   └── trainer.py           # Training logic
│   ├── data/                    # Xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── processor.py         # Character processor
│   │   └── dataset.py           # PyTorch Dataset
│   ├── training/                # Training pipeline
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration
│   │   └── pipeline.py          # Training pipeline
│   └── __init__.py
├── scripts/                     # Scripts sử dụng
│   ├── train.py                 # Training script
│   ├── test.py                  # Testing script
│   └── inference.py             # Inference script
├── configs/                     # Configuration files
├── data/                        # Dữ liệu
│   ├── corpus_splitted/         # Corpus đã phân tách
│   └── ngrams/                  # N-gram dictionaries
├── outputs/                     # Kết quả training
└── README.md
```

## Tính Năng Chính

### Mô Hình A-TCN

- **Acausal Dilated Convolutions**: Sử dụng context hai chiều (quá khứ + tương lai)
- **Character-level Processing**: Xử lý ở mức ký tự
- **Large Receptive Field**: Receptive field lên đến 127 ký tự
- **Efficient Architecture**: ~2.5M parameters, ~10MB model size

### Xử Lý Dữ Liệu

- Vocabulary: 233 ký tự tiếng Việt (bao gồm có dấu + không dấu)
- Character tokenization với attention masks
- Streaming data loading cho corpus lớn
- Train/validation split tự động

### Training Pipeline

- Cấu hình linh hoạt với presets (small/default/large)
- Early stopping và learning rate scheduling
- Checkpointing và model saving
- Progress tracking với tqdm
- Sample prediction testing

## Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.8+
- PyTorch 1.9+
- CUDA (tùy chọn, cho GPU training)

### Cài Đặt Dependencies

```bash
pip install torch torchvision torchaudio
pip install tqdm numpy
```

## Sử Dụng

### 1. Training

#### Training Cơ Bản

```bash
python scripts/train.py
```

#### Training với Cấu Hình Tùy Chỉnh

```bash
# Sử dụng preset nhỏ cho testing nhanh
python scripts/train.py --config small --num_epochs 10

# Sử dụng preset lớn cho performance tốt nhất
python scripts/train.py --config large --batch_size 64

# Tùy chỉnh chi tiết
python scripts/train.py \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --num_layers 6 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --num_epochs 50 \
    --max_files 10 \
    --output_dir outputs/experiment_1
```

#### Training với Data Giới Hạn (cho testing)

```bash
python scripts/train.py \
    --config small \
    --max_files 5 \
    --max_samples_per_file 1000 \
    --num_epochs 5
```

### 2. Testing

#### Test Model từ Checkpoint

```bash
python scripts/test.py outputs/best_model.pt
```

#### Test Interactive

```bash
python scripts/test.py outputs/best_model.pt --interactive
```

#### Test với Custom Input

```bash
python scripts/test.py outputs/best_model.pt \
    --test_inputs "toi yeu viet nam" "chung ta la mot" "hoc tap tot"
```

### 3. Inference (Production)

#### Interactive Mode

```bash
python scripts/inference.py outputs/best_model.pt --interactive
```

#### Xử lý Single Text

```bash
python scripts/inference.py outputs/best_model.pt \
    --text "toi yeu viet nam"
```

#### Xử lý File

```bash
python scripts/inference.py outputs/best_model.pt \
    --input_file input.txt \
    --output_file output.txt
```

#### Xử lý JSON File

```bash
python scripts/inference.py outputs/best_model.pt \
    --input_file data.json \
    --output_file output.json \
    --input_format json \
    --input_key "text" \
    --output_key "restored_text"
```

## API Sử Dụng

### Python API

```python
from src.models import ATCN, ATCNTrainer
from src.data import VietnameseCharProcessor
from src.training import ATCNTrainingPipeline, TrainingConfig

# Tạo config
config = TrainingConfig(
    embedding_dim=128,
    hidden_dim=256,
    num_layers=6,
    batch_size=32
)

# Training
pipeline = ATCNTrainingPipeline(config)
history = pipeline.run()

# Inference
from scripts.inference import VietnameseAccentRestorer
restorer = VietnameseAccentRestorer('outputs/best_model.pt')
result = restorer.restore_accents("toi yeu viet nam")
print(result)  # "tôi yêu việt nam"
```

## Cấu Hình (Configuration)

### Presets Có Sẵn

#### Small (cho testing nhanh)

- embedding_dim: 64
- hidden_dim: 128
- num_layers: 4
- batch_size: 16
- max_files: 10

#### Default (cân bằng)

- embedding_dim: 128
- hidden_dim: 256
- num_layers: 6
- batch_size: 32

#### Large (performance tối ưu)

- embedding_dim: 256
- hidden_dim: 512
- num_layers: 8
- batch_size: 64
- max_dilation: 64

### Tùy Chỉnh Configuration

```python
from src.training import TrainingConfig

config = TrainingConfig(
    # Model parameters
    embedding_dim=128,
    hidden_dim=256,
    num_layers=6,

    # Training parameters
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=50,

    # Data parameters
    corpus_dir="corpus_splitted",
    max_files=None,  # Tất cả files
    max_samples_per_file=10000,

    # Output
    output_dir="outputs"
)
```

## Kết Quả Training

Training sẽ tạo ra:

- `outputs/best_model.pt`: Model tốt nhất
- `outputs/config.json`: Configuration đã sử dụng
- `outputs/vocab.json`: Vocabulary mapping
- `outputs/training_results.json`: Lịch sử training
- `outputs/checkpoint_epoch_N.pt`: Checkpoints theo epochs

## Monitoring Training

Training progress được track qua:

- Training/validation loss và accuracy
- Learning rate scheduling
- Sample predictions mỗi 10 epochs
- Early stopping khi overfitting

## Ví Dụ Kết Quả

```
Input:  "toi yeu viet nam"
Output: "tôi yêu việt nam"

Input:  "chung ta la mot"
Output: "chúng ta là một"

Input:  "hoc tap cham chi"
Output: "học tập chăm chỉ"
```

## Khắc Phục Sự Cố

### Out of Memory

```bash
# Giảm batch size
python scripts/train.py --batch_size 16

# Sử dụng CPU
python scripts/train.py --device cpu

# Giảm model size
python scripts/train.py --config small
```

### Training Chậm

```bash
# Giới hạn data cho testing
python scripts/train.py --max_files 10 --max_samples_per_file 1000

# Sử dụng GPU
python scripts/train.py --device cuda
```

### Model Không Converge

```bash
# Tăng learning rate
python scripts/train.py --learning_rate 5e-3

# Tăng model capacity
python scripts/train.py --config large

# Tăng epochs
python scripts/train.py --num_epochs 100
```

## Kiến Trúc Chi Tiết

### A-TCN Model

- Character embedding (vocab_size=233)
- 6 TCN residual blocks với dilations [1,2,4,8,16,32]
- Acausal convolutions (bidirectional context)
- Output projection cho character prediction

### Training Features

- Character-level cross-entropy loss
- AdamW optimizer với weight decay
- ReduceLROnPlateau scheduling
- Gradient clipping
- Early stopping

## Tương Lai

- [ ] Beam search decoding
- [ ] Attention mechanisms
- [ ] Multi-task learning
- [ ] Web API deployment
- [ ] Mobile optimization
