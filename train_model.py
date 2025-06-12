import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import json
from model_architecture import VietnameseAccentRestorer
from data_preprocessing import VietnameseDataPreprocessor

class AccentRestorationDataset(Dataset):
    """
    Dataset cho bài toán phục hồi dấu tiếng Việt
    """
    
    def __init__(self, input_sequences, target_sequences, char_to_idx, max_length=512):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences
        self.char_to_idx = char_to_idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, idx):
        input_text = self.input_sequences[idx]
        target_text = self.target_sequences[idx]
        
        # Cắt ngắn nếu quá dài
        if len(input_text) > self.max_length:
            input_text = input_text[:self.max_length]
            target_text = target_text[:self.max_length]
        
        # Encode thành indices
        input_ids = [self.char_to_idx.get(char, 0) for char in input_text]
        target_ids = [self.char_to_idx.get(char, 0) for char in target_text]
        
        # Padding
        input_ids += [0] * (self.max_length - len(input_ids))
        target_ids += [0] * (self.max_length - len(target_ids))
        
        return {
            'input_ids': torch.tensor(input_ids[:self.max_length], dtype=torch.long),
            'target_ids': torch.tensor(target_ids[:self.max_length], dtype=torch.long),
            'length': torch.tensor(len(input_text), dtype=torch.long)
        }

class AccentRestorationTrainer:
    """
    Trainer cho mô hình phục hồi dấu tiếng Việt
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.model.to(device)  # model.model là ACausalTCN
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.model.parameters(),  # model.model.parameters()
            lr=1e-3,
            weight_decay=1e-4
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            factor=0.5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def calculate_accuracy(self, predictions, targets, lengths):
        """
        Tính độ chính xác character-level
        """
        predictions = predictions.argmax(dim=-1)
        correct = 0
        total = 0
        
        for i, length in enumerate(lengths):
            pred = predictions[i][:length]
            target = targets[i][:length]
            correct += (pred == target).sum().item()
            total += length.item()
        
        return correct / total if total > 0 else 0
    
    def train_epoch(self, train_loader):
        """
        Huấn luyện một epoch
        """
        self.model.model.train()  # model.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            lengths = batch['length']
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model.model(input_ids)
            
            # Reshape cho loss calculation
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = target_ids.view(-1)
            
            # Calculate loss
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Calculate accuracy
            acc = self.calculate_accuracy(outputs, target_ids, lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        return avg_loss, avg_acc
    
    def validate_epoch(self, val_loader):
        """
        Validation một epoch
        """
        self.model.model.eval()  # model.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                lengths = batch['length']
                
                # Forward pass
                outputs = self.model.model(input_ids)
                
                # Reshape cho loss calculation
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = target_ids.view(-1)
                
                # Calculate loss
                loss = self.criterion(outputs_flat, targets_flat)
                
                # Calculate accuracy
                acc = self.calculate_accuracy(outputs, target_ids, lengths)
                
                total_loss += loss.item()
                total_acc += acc
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        return avg_loss, avg_acc
    
    def train(self, train_loader, val_loader, num_epochs, save_dir="models"):
        """
        Huấn luyện mô hình
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        print(f"Bắt đầu huấn luyện trong {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Số tham số: {sum(p.numel() for p in self.model.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, "best_model.pth")
                # Đảm bảo thư mục tồn tại trước khi save
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                self.model.save_model(best_model_path)
                print(f"Đã lưu best model tại: {best_model_path}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            # Đảm bảo thư mục tồn tại trước khi save
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'history': self.history,
                'val_loss': val_loss
            }, checkpoint_path)
        
        # Save training history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nHuấn luyện hoàn thành!")
        print(f"Best validation loss: {best_val_loss:.4f}")

def load_training_data(data_file):
    """
    Tải dữ liệu huấn luyện
    """
    input_texts = []
    target_texts = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                no_accent, with_accent = line.split('\t', 1)
                input_texts.append(no_accent)
                target_texts.append(with_accent)
    
    return input_texts, target_texts

def main():
    """
    Hàm chính để huấn luyện mô hình
    """
    # Cấu hình
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")
    
    # Load dữ liệu
    data_file = "data/training_pairs.txt"
    
    if not os.path.exists(data_file):
        print("Tạo dữ liệu huấn luyện...")
        from data_preprocessing import main as preprocess_main
        preprocess_main()
    
    print("Tải dữ liệu huấn luyện...")
    input_texts, target_texts = load_training_data(data_file)
    print(f"Đã tải {len(input_texts)} mẫu dữ liệu")
    
    # Khởi tạo mô hình
    model = VietnameseAccentRestorer()
    
    # Chia dữ liệu train/val
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        input_texts, target_texts, test_size=0.1, random_state=42
    )
    
    print(f"Train: {len(train_inputs)} samples")
    print(f"Validation: {len(val_inputs)} samples")
    
    # Tạo datasets
    max_length = 256  # Giảm để tiết kiệm memory
    
    train_dataset = AccentRestorationDataset(
        train_inputs, train_targets, model.char_to_idx, max_length
    )
    val_dataset = AccentRestorationDataset(
        val_inputs, val_targets, model.char_to_idx, max_length
    )
    
    # Tạo data loaders
    batch_size = 32 if device.type == 'cuda' else 16
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0
    )
    
    # Khởi tạo trainer
    trainer = AccentRestorationTrainer(model, device)
    
    # Huấn luyện
    num_epochs = 20
    trainer.train(train_loader, val_loader, num_epochs)
    
    # Export ONNX
    print("\nXuất mô hình sang ONNX...")
    onnx_path = "models/accent_restorer.onnx"
    model.export_to_onnx(onnx_path, max_seq_len=max_length)
    
    print(f"Mô hình đã được lưu và export thành công!")

if __name__ == "__main__":
    main() 