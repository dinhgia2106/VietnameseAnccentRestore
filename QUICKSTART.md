# Quick Start - Vietnamese Accent Restoration

## Cài đặt (30 giây)

```bash
pip install -r requirements.txt
python test.py  # Verify installation
```

## Sử dụng ngay (Copy & Paste)

### Basic Usage

```python
from vietnamese_accent_restore import VietnameseAccentRestore

restore = VietnameseAccentRestore()
suggestions = restore.find_suggestions("may bay")
print(suggestions[0][0])  # "máy bay"
```

### Advanced Usage (Hybrid System)

```python
from integrated_system import IntegratedAccentSystem

system = IntegratedAccentSystem()
system.initialize_components()

predictions = system.predict_hybrid("may bay khong nguoi lai")
print(predictions[0].text)  # "máy bay không người lái"
```

## Training A-TCN (có corpus)

```bash
# 1. Create training data
python train_atcn.py --create_data --corpus_file your_corpus.txt

# 2. Train model
python train_atcn.py --epochs 20

# 3. Test
python train_atcn.py --test_text "may bay khong nguoi lai"
```

## Demo Full Workflow

```bash
python demo_full_training.py
```

Xem [README.md](README.md) để biết chi tiết.

## Files chính

- `vietnamese_accent_restore.py` - N-gram system
- `integrated_system.py` - Hybrid system (N-gram + A-TCN)
- `train_atcn.py` - Training pipeline A-TCN
- `config.py` - Configuration
- `test.py` - System testing
