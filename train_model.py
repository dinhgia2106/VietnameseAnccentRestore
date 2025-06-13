import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import json
from model_architecture import VietnameseAccentRestorer

class TwoPhaseTrainer:
    """
    Trainer 2 phase cho việc học Vietnamese accent restoration
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.model.to(device)
        
        # Simple loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Training history
        self.history = {
            'phase1_train_loss': [],
            'phase1_val_loss': [],
            'phase1_train_acc': [],
            'phase1_val_acc': [],
            'phase2_train_loss': [],
            'phase2_val_loss': [],
            'phase2_train_acc': [],
            'phase2_val_acc': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience = 7  # Tăng patience vì 2-phase training
        self.patience_counter = 0
    
    def create_optimizer(self, lr=2e-3):
        """Tạo optimizer mới cho mỗi phase"""
        return optim.AdamW(
            self.model.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            betas=(0.9, 0.95)
        )
    
    def create_scheduler(self, optimizer, T_max=15):
        """Tạo scheduler mới cho mỗi phase"""
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=1e-5
        )
    
    def calculate_accuracy(self, predictions, targets, lengths):
        """Tính accuracy nhanh"""
        predictions = predictions.argmax(dim=-1)
        correct = 0
        total = 0
        
        for i, length in enumerate(lengths):
            pred = predictions[i][:length]
            target = targets[i][:length]
            mask = target != 0
            if mask.sum() > 0:
                correct += ((pred == target) * mask).sum().item()
                total += mask.sum().item()
        
        return correct / total if total > 0 else 0
    
    def train_epoch(self, train_loader, optimizer):
        """Training epoch"""
        self.model.model.train()
        total_loss = 0
        total_acc = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            lengths = batch['length']
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model.model(input_ids)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = target_ids.view(-1)
            
            # Loss
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Accuracy
            acc = self.calculate_accuracy(outputs, target_ids, lengths)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc
        
        return total_loss / len(train_loader), total_acc / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """Validation epoch"""
        self.model.model.eval()
        total_loss = 0
        total_acc = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                lengths = batch['length']
                
                outputs = self.model.model(input_ids)
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = target_ids.view(-1)
                
                loss = self.criterion(outputs_flat, targets_flat)
                acc = self.calculate_accuracy(outputs, target_ids, lengths)
                
                total_loss += loss.item()
                total_acc += acc
        
        return total_loss / len(val_loader), total_acc / len(val_loader)
    
    def train_phase1(self, train_loader, val_loader, num_epochs, save_dir="models"):
        """Phase 1: Train với full Viet74K data"""
        print("=== PHASE 1: Training với full Viet74K data ===")
        
        optimizer = self.create_optimizer(lr=2e-3)
        scheduler = self.create_scheduler(optimizer, T_max=num_epochs)
        
        for epoch in range(num_epochs):
            print(f"\nPhase 1 - Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            
            # Validation  
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            self.history['phase1_train_loss'].append(train_loss)
            self.history['phase1_val_loss'].append(val_loss)
            self.history['phase1_train_acc'].append(train_acc)
            self.history['phase1_val_acc'].append(val_acc)
            
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                best_path = os.path.join(save_dir, "phase1_best_model.pth")
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                self.model.save_model(best_path)
                print(f"Saved Phase 1 best: {best_path}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"Early stopping sau {epoch+1} epochs")
                break
        
        print("Phase 1 hoàn thành!")
    
    def train_phase2(self, all_data, val_loader, num_epochs, samples_per_epoch=50000, save_dir="models"):
        """Phase 2: Mỗi epoch sample 50k từ toàn bộ data"""
        print("\n=== PHASE 2: Training với sampling mỗi epoch ===")
        
        # Reset early stopping cho Phase 2  
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        phase2_patience = 8  # Patience cao hơn cho Phase 2
        print(f"Reset early stopping cho Phase 2 (patience: {phase2_patience})")
        
        optimizer = self.create_optimizer(lr=1e-3)  # LR thấp hơn cho phase 2
        scheduler = self.create_scheduler(optimizer, T_max=num_epochs)
        
        input_texts, target_texts = all_data
        total_samples = len(input_texts)
        print(f"Tổng data: {total_samples} samples, mỗi epoch: {samples_per_epoch} samples")
        
        for epoch in range(num_epochs):
            print(f"\nPhase 2 - Epoch {epoch+1}/{num_epochs}")
            
            # Random sampling 50k samples
            import random
            indices = random.sample(range(total_samples), min(samples_per_epoch, total_samples))
            epoch_inputs = [input_texts[i] for i in indices]
            epoch_targets = [target_texts[i] for i in indices]
            
            # Tạo dataset cho epoch này
            epoch_dataset = AccentDataset(
                epoch_inputs, epoch_targets, self.model.char_to_idx, max_length=128
            )
            
            batch_size = 128 if self.device.type == 'cuda' else 64
            epoch_loader = DataLoader(
                epoch_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch_loader, optimizer)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            self.history['phase2_train_loss'].append(train_loss)
            self.history['phase2_val_loss'].append(val_loss)
            self.history['phase2_train_acc'].append(train_acc)
            self.history['phase2_val_acc'].append(val_acc)
            
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint mỗi epoch
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f"phase2_epoch_{epoch+1}.pth")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                self.model.save_model(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                best_path = os.path.join(save_dir, "phase2_best_model.pth")
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                self.model.save_model(best_path)
                print(f"Saved Phase 2 best: {best_path}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= phase2_patience:
                print(f"Early stopping sau {epoch+1} epochs")
                break
        
        print("Phase 2 hoàn thành!")
    
    def train_two_phase(self, viet74k_data, corpus_data, phase1_epochs=10, phase2_epochs=30, save_dir="models"):
        """Training 2 phase complete"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("Bắt đầu 2-Phase Training!")
        print(f"Device: {self.device}")
        print(f"Tham số: {sum(p.numel() for p in self.model.model.parameters()):,}")
        
        # Chuẩn bị data Phase 1
        input_texts, target_texts = viet74k_data
        
        # Split validation set 10k
        val_size = min(10000, len(input_texts) // 10)
        train_inputs = input_texts[val_size:]
        train_targets = target_texts[val_size:]
        val_inputs = input_texts[:val_size]
        val_targets = target_texts[:val_size]
        
        print(f"Phase 1 - Train: {len(train_inputs)}, Val: {len(val_inputs)}")
        
        # Dataset Phase 1
        train_dataset = AccentDataset(train_inputs, train_targets, self.model.char_to_idx, 128)
        val_dataset = AccentDataset(val_inputs, val_targets, self.model.char_to_idx, 128)
        
        batch_size = 128 if self.device.type == 'cuda' else 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Phase 1 Training
        self.train_phase1(train_loader, val_loader, phase1_epochs, save_dir)
        
        # Phase 2 Training với corpus data
        if corpus_data:
            self.train_phase2(corpus_data, val_loader, phase2_epochs, save_dir=save_dir)
        
        print("2-Phase Training hoàn thành!")

class AccentRestorationTrainer:
    """
     Trainer tối ưu cho tốc độ training
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.model.to(device)
        
        # Simple loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        #  optimizer với higher learning rate
        self.optimizer = optim.AdamW(
            model.model.parameters(),
            lr=2e-3,  # Tăng LR để converge nhanh
            weight_decay=1e-4,
            betas=(0.9, 0.95)  # er momentum
        )
        
        # Simple scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=15, eta_min=1e-5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 3  # Giảm patience để stop sớm hơn
        self.patience_counter = 0
    
    def calculate_accuracy(self, predictions, targets, lengths):
        """Tính accuracy nhanh"""
        predictions = predictions.argmax(dim=-1)
        correct = 0
        total = 0
        
        for i, length in enumerate(lengths):
            pred = predictions[i][:length]
            target = targets[i][:length]
            mask = target != 0
            if mask.sum() > 0:
                correct += ((pred == target) * mask).sum().item()
                total += mask.sum().item()
        
        return correct / total if total > 0 else 0
    
    def train_epoch(self, train_loader):
        """Training epoch nhanh"""
        self.model.model.train()
        total_loss = 0
        total_acc = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            lengths = batch['length']
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model.model(input_ids)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = target_ids.view(-1)
            
            # Loss
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Accuracy
            acc = self.calculate_accuracy(outputs, target_ids, lengths)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc
        
        return total_loss / len(train_loader), total_acc / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """Validation nhanh"""
        self.model.model.eval()
        total_loss = 0
        total_acc = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                lengths = batch['length']
                
                outputs = self.model.model(input_ids)
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = target_ids.view(-1)
                
                loss = self.criterion(outputs_flat, targets_flat)
                acc = self.calculate_accuracy(outputs, target_ids, lengths)
                
                total_loss += loss.item()
                total_acc += acc
        
        return total_loss / len(val_loader), total_acc / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs, save_dir="models"):
        """ training với early stopping"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f" Training - {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Tham số: {sum(p.numel() for p in self.model.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                best_path = os.path.join(save_dir, "best_model.pth")
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                self.model.save_model(best_path)
                print(f"Saved best: {best_path}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"Early stopping sau {epoch+1} epochs")
                break
        
        print(f"Training hoàn thành! Best val loss: {self.best_val_loss:.4f}")

class AccentDataset(Dataset):
    """Dataset tối ưu cho tốc độ"""
    
    def __init__(self, input_texts, target_texts, char_to_idx, max_length=128):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.char_to_idx = char_to_idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.input_texts)
    
    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        
        # Truncate nếu quá dài
        if len(input_text) > self.max_length:
            input_text = input_text[:self.max_length]
            target_text = target_text[:self.max_length]
        
        # Encode
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

def load_training_data(data_file):
    """Load data nhanh"""
    input_texts = []
    target_texts = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                no_accent, with_accent = line.split('\t', 1)
                # Chỉ lấy câu ngắn để training nhanh
                if 5 <= len(no_accent) <= 100:
                    input_texts.append(no_accent)
                    target_texts.append(with_accent)
    
    return input_texts, target_texts

def load_viet74k_data(data_file):
    """Load data từ Viet74K.txt"""
    from unidecode import unidecode
    input_texts = []
    target_texts = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line) >= 2:  # Lọc từ có ít nhất 2 ký tự
                # Tạo cặp (không dấu, có dấu)
                with_accent = line
                no_accent = unidecode(with_accent)
                
                # Chỉ thêm nếu khác nhau (có dấu)
                if no_accent != with_accent and 2 <= len(with_accent) <= 50:
                    input_texts.append(no_accent)
                    target_texts.append(with_accent)
    
    return input_texts, target_texts

def main():
    """ 2-Phase training main"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Viet74K data cho Phase 1
    viet74k_file = "data/Viet74K.txt"
    if not os.path.exists(viet74k_file):
        print(f"Không tìm thấy {viet74k_file}")
        return
        
    print("Loading Viet74K data...")
    viet74k_inputs, viet74k_targets = load_viet74k_data(viet74k_file)
    print(f"Viet74K data: {len(viet74k_inputs)} samples")
    
    # Load corpus data cho Phase 2 (nếu có)
    corpus_data = None
    corpus_file = "data/training_pairs.txt"
    if os.path.exists(corpus_file):
        print("Loading corpus data cho Phase 2...")
        corpus_inputs, corpus_targets = load_training_data(corpus_file)
        corpus_data = (corpus_inputs, corpus_targets)
        print(f"Corpus data: {len(corpus_inputs)} samples")
    else:
        print("Tạo dữ liệu corpus...")
        try:
            from data_preprocessing import main as preprocess_main
            preprocess_main()
            corpus_inputs, corpus_targets = load_training_data(corpus_file)
            corpus_data = (corpus_inputs, corpus_targets)
            print(f"Corpus data: {len(corpus_inputs)} samples")
        except:
            print("Không thể tạo corpus data, chỉ chạy Phase 1")
    
    # Model
    model = VietnameseAccentRestorer(use_enhanced_model=False)
    
    # Two-phase trainer
    trainer = TwoPhaseTrainer(model, device)
    
    # Training config
    phase1_epochs = 15  # Phase 1: học full Viet74K
    phase2_epochs = 30  # Phase 2: sampling epoch
    
    # Bắt đầu 2-phase training
    trainer.train_two_phase(
        viet74k_data=(viet74k_inputs, viet74k_targets),
        corpus_data=corpus_data,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        save_dir="models"
    )
    
    print("2-Phase Training hoàn thành!")

if __name__ == "__main__":
    main() 