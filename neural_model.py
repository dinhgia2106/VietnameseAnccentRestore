import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pickle
import os
import time
import re
from collections import defaultdict
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import heapq

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc="", disable=False):
        if desc:
            print(f"{desc}...")
        return iterable

class CharacterDataset(Dataset):
    """Dataset cho character-level sequence labeling"""
    def __init__(self, sequences, labels, char_to_idx_input, char_to_idx_output, max_len=64):
        self.sequences = sequences
        self.labels = labels
        self.char_to_idx_input = char_to_idx_input
        self.char_to_idx_output = char_to_idx_output
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx][:self.max_len]
        label = self.labels[idx][:self.max_len]
        
        # Convert to indices
        seq_indices = [self.char_to_idx_input.get(c, 1) for c in seq]  # 1 = UNK
        label_indices = [self.char_to_idx_output.get(c, 1) for c in label]
        
        # Pad to max_len
        while len(seq_indices) < self.max_len:
            seq_indices.append(0)  # 0 = PAD
            label_indices.append(0)
        
        return torch.tensor(seq_indices), torch.tensor(label_indices), len(seq)

class ToneRestorationModel(nn.Module):
    """Mô hình Bi-GRU được tối ưu với các tùy chọn kích thước"""
    def __init__(self, input_vocab_size, output_vocab_size, 
                 embedding_dim=64, hidden_dim=128, dropout_rate=0.3, 
                 lightweight=False):
        super().__init__()
        
        # Tùy chọn lightweight để giảm kích thước model
        if lightweight:
            embedding_dim = min(embedding_dim, 32)
            hidden_dim = min(hidden_dim, 64)
            dropout_rate = max(dropout_rate, 0.4)  # Tăng dropout cho model nhỏ
        
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=0)
        
        # Layer normalization cho embedding
        self.embedding_ln = nn.LayerNorm(embedding_dim)
        
        # Bi-GRU với dropout
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, 
                         bidirectional=True, dropout=dropout_rate if hidden_dim > 1 else 0)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers với residual connection
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_vocab_size)
        
        # Layer normalization cho fc
        self.fc_ln = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Khởi tạo weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        # Embedding với layer norm
        embedded = self.embedding(x)
        embedded = self.embedding_ln(embedded)
        
        # GRU
        gru_out, _ = self.gru(embedded)
        
        # Batch normalization (reshape cho batch norm)
        batch_size, seq_len, hidden_size = gru_out.shape
        gru_out_reshaped = gru_out.reshape(batch_size * seq_len, hidden_size)
        gru_out_normalized = self.batch_norm(gru_out_reshaped)
        gru_out = gru_out_normalized.reshape(batch_size, seq_len, hidden_size)
        
        # Dropout
        output = self.dropout(gru_out)
        
        # FC layers với residual connection
        fc1_out = torch.relu(self.fc1(output))
        fc1_out = self.fc_ln(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        # Output
        logits = self.fc2(fc1_out)
        
        return logits

class SentenceDataset(Dataset):
    """Dataset cho sentence-level training"""
    def __init__(self, sentences_no_tone, sentences_with_tone, 
                 char_to_idx_input, char_to_idx_output, max_len=128):
        self.sentences_no_tone = sentences_no_tone
        self.sentences_with_tone = sentences_with_tone
        self.char_to_idx_input = char_to_idx_input
        self.char_to_idx_output = char_to_idx_output
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sentences_no_tone)
    
    def __getitem__(self, idx):
        seq = self.sentences_no_tone[idx][:self.max_len]
        label = self.sentences_with_tone[idx][:self.max_len]
        
        # Convert to indices
        seq_indices = [self.char_to_idx_input.get(c, 1) for c in seq]  # 1 = UNK
        label_indices = [self.char_to_idx_output.get(c, 1) for c in label]
        
        # Pad to max_len
        while len(seq_indices) < self.max_len:
            seq_indices.append(0)  # 0 = PAD
            label_indices.append(0)
        
        return torch.tensor(seq_indices), torch.tensor(label_indices), len(seq)

class BeamSearchNode:
    """Node cho beam search"""
    def __init__(self, sequence="", score=0.0, position=0):
        self.sequence = sequence
        self.score = score
        self.position = position
    
    def __lt__(self, other):
        return self.score > other.score  # Higher score = better

class EarlyStopping:
    """Early stopping để tránh overfitting"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def save_checkpoint(self, model):
        """Lưu best weights"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
    
    def restore_weights(self, model):
        """Khôi phục best weights"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

class NeuralToneModel:
    def __init__(self, model_dir="models/neural", lightweight=False):
        self.model_dir = model_dir
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lightweight = lightweight
        
        # Tone mappings
        self.tone_mapping = {
            'a': ['a', 'á', 'à', 'ả', 'ã', 'ạ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ'],
            'e': ['e', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ'],
            'i': ['i', 'í', 'ì', 'ỉ', 'ĩ', 'ị'],
            'o': ['o', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ'],
            'u': ['u', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự'],
            'y': ['y', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ'],
            'd': ['d', 'đ']
        }
        
        self.remove_tone_map = {}
        for base, variants in self.tone_mapping.items():
            for variant in variants:
                self.remove_tone_map[variant] = base
        
        # Special tokens for sentence processing
        self.WORD_SEPARATOR = ' '
        self.SENTENCE_START = '<START>'
        self.SENTENCE_END = '<END>'
        
    def remove_tones(self, text):
        """Loại bỏ dấu khỏi text"""
        return ''.join(self.remove_tone_map.get(c, c) for c in text.lower())
    
    def prepare_data(self, dict_path, corpus_path, max_samples=100000, use_sentences=True):
        """Chuẩn bị dữ liệu training với tùy chọn sentence-level"""
        print("Chuẩn bị dữ liệu training...")
        
        sequences_no_tone = []
        sequences_with_tone = []
        
        if use_sentences:
            print("Sử dụng sentence-level training...")
            return self._prepare_sentence_data(corpus_path, max_samples)
        else:
            print("Sử dụng word-level training...")
            return self._prepare_word_data(dict_path, corpus_path, max_samples)
    
    def _prepare_sentence_data(self, corpus_path, max_samples):
        """Chuẩn bị dữ liệu ở mức câu"""
        sentences_no_tone = []
        sentences_with_tone = []
        
        count = 0
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break
                
                line = line.strip()
                if len(line) > 10 and len(line) < 200:  # Lọc câu có độ dài hợp lý
                    # Làm sạch câu
                    sentence_with_tone = re.sub(r'[^\w\s]', '', line.lower())
                    sentence_no_tone = self.remove_tones(sentence_with_tone)
                    
                    if sentence_no_tone != sentence_with_tone:  # Chỉ lấy câu có dấu
                        sentences_no_tone.append(sentence_no_tone)
                        sentences_with_tone.append(sentence_with_tone)
                        count += 1
        
        print(f"Sentence data: {len(sentences_no_tone)} câu")
        
        # Tạo vocabularies từ sentence data
        chars_input = set()
        chars_output = set()
        
        for sentence in sentences_no_tone:
            chars_input.update(sentence)
        for sentence in sentences_with_tone:
            chars_output.update(sentence)
        
        # Vocab mapping
        self.char_to_idx_input = {'<PAD>': 0, '<UNK>': 1}
        for i, char in enumerate(sorted(chars_input), 2):
            self.char_to_idx_input[char] = i
        
        self.char_to_idx_output = {'<PAD>': 0, '<UNK>': 1}
        for i, char in enumerate(sorted(chars_output), 2):
            self.char_to_idx_output[char] = i
        
        self.idx_to_char_output = {v: k for k, v in self.char_to_idx_output.items()}
        
        print(f"Input vocab: {len(self.char_to_idx_input)} chars")
        print(f"Output vocab: {len(self.char_to_idx_output)} chars")
        
        return sentences_no_tone, sentences_with_tone
    
    def _prepare_word_data(self, dict_path, corpus_path, max_samples):
        """Chuẩn bị dữ liệu ở mức từ (code gốc)"""
        sequences_no_tone = []
        sequences_with_tone = []
        
        # Từ dictionary
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and len(word) <= 20:
                    no_tone = self.remove_tones(word)
                    if no_tone != word:
                        sequences_no_tone.append(no_tone)
                        sequences_with_tone.append(word)
        
        print(f"Dictionary: {len(sequences_no_tone)} từ")
        
        # Từ corpus
        count = 0
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples is not None and count >= max_samples:
                    break
                    
                words = re.findall(r'\w+', line.strip().lower())
                for word in words:
                    if len(word) <= 20 and len(word) > 1:
                        no_tone = self.remove_tones(word)
                        if no_tone != word:
                            sequences_no_tone.append(no_tone)
                            sequences_with_tone.append(word)
                            count += 1
                            if max_samples is not None and count >= max_samples:
                                break
        
        print(f"Tổng cộng: {len(sequences_no_tone)} sequences")
        
        # Tạo vocabularies
        chars_input = set()
        chars_output = set()
        
        for seq in sequences_no_tone:
            chars_input.update(seq)
        for seq in sequences_with_tone:
            chars_output.update(seq)
        
        # Vocab mapping
        self.char_to_idx_input = {'<PAD>': 0, '<UNK>': 1}
        for i, char in enumerate(sorted(chars_input), 2):
            self.char_to_idx_input[char] = i
        
        self.char_to_idx_output = {'<PAD>': 0, '<UNK>': 1}
        for i, char in enumerate(sorted(chars_output), 2):
            self.char_to_idx_output[char] = i
        
        self.idx_to_char_output = {v: k for k, v in self.char_to_idx_output.items()}
        
        print(f"Input vocab: {len(self.char_to_idx_input)} chars")
        print(f"Output vocab: {len(self.char_to_idx_output)} chars")
        
        return sequences_no_tone, sequences_with_tone
    
    def train(self, dict_path, corpus_path, epochs=100, batch_size=64, 
              learning_rate=0.001, weight_decay=1e-4, use_sentences=True):
        """Training model với sentence-level processing"""
        mode_str = "sentence-level" if use_sentences else "word-level"
        weight_str = "lightweight" if self.lightweight else "standard"
        print(f"Bắt đầu training neural model - {mode_str} - {weight_str}")
        print(f"Epochs: {epochs}, L2 weight decay: {weight_decay}")
        
        # Chuẩn bị data
        max_samples = 10000000 if use_sentences else 2000000
        seq_input, seq_output = self.prepare_data(dict_path, corpus_path, 
                                                 max_samples=max_samples, 
                                                 use_sentences=use_sentences)
        
        # Split data
        train_input, val_input, train_output, val_output = train_test_split(
            seq_input, seq_output, test_size=0.15, random_state=42
        )
        
        # Datasets với max_len phù hợp
        max_len = 128 if use_sentences else 64
        if use_sentences:
            train_dataset = SentenceDataset(train_input, train_output, 
                                          self.char_to_idx_input, self.char_to_idx_output, max_len)
            val_dataset = SentenceDataset(val_input, val_output,
                                        self.char_to_idx_input, self.char_to_idx_output, max_len)
        else:
            train_dataset = CharacterDataset(train_input, train_output, 
                                           self.char_to_idx_input, self.char_to_idx_output, max_len)
            val_dataset = CharacterDataset(val_input, val_output,
                                         self.char_to_idx_input, self.char_to_idx_output, max_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Model với tùy chọn lightweight
        input_vocab_size = len(self.char_to_idx_input)
        output_vocab_size = len(self.char_to_idx_output)
        
        if self.lightweight:
            embedding_dim, hidden_dim = 32, 64
        else:
            embedding_dim, hidden_dim = 128, 256
        
        self.model = ToneRestorationModel(input_vocab_size, output_vocab_size, 
                                        embedding_dim=embedding_dim, 
                                        hidden_dim=hidden_dim,
                                        dropout_rate=0.2,
                                        lightweight=self.lightweight).to(self.device)
        
        # Loss và optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, 
                             weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, 
                                    patience=3, verbose=True, min_lr=1e-5)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {param_count:,}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not TQDM_AVAILABLE)
            for batch_idx, (inputs, targets, lengths) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                # Accuracy
                pred = outputs.argmax(dim=-1)
                mask = targets != 0  # Không tính PAD
                train_correct += (pred == targets)[mask].sum().item()
                train_total += mask.sum().item()
                
                if batch_idx % 100 == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                                    'acc': f'{train_correct/train_total:.4f}',
                                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets, lengths in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                    val_loss += loss.item()
                    
                    pred = outputs.argmax(dim=-1)
                    mask = targets != 0
                    val_correct += (pred == targets)[mask].sum().item()
                    val_total += mask.sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"           Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check if this is the best model
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                print(f"Tìm thấy model tốt hơn với val_loss: {avg_val_loss:.4f}")
            
            # Lưu checkpoint
            self.save_model(epoch=epoch+1, val_loss=avg_val_loss, is_best=is_best, 
                          mode=mode_str)
            
            # Cleanup old checkpoints
            if (epoch + 1) % 3 == 0:
                self.cleanup_old_checkpoints(keep_last=5)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if early_stopping(avg_val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                early_stopping.restore_weights(self.model)
                self.save_model(epoch=epoch+1, val_loss=avg_val_loss, is_best=True, 
                              mode=mode_str)
                break
        
        # Save final model
        self.save_model(epoch=epochs, val_loss=val_losses[-1] if val_losses else 0.0, 
                       mode=mode_str)
        print("Training completed!")
        
        # Final cleanup
        self.cleanup_old_checkpoints(keep_last=3)
        
        # Training summary
        print(f"\nTraining Summary:")
        print(f"Mode: {mode_str}")
        print(f"Model: {weight_str}")
        print(f"Parameters: {param_count:,}")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final val loss: {val_losses[-1]:.4f}")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Model được lưu tại: {self.model_dir}")
    
    def predict_sentence_beam_search(self, sentence, beam_width=5, max_length=128):
        """Dự đoán câu sử dụng beam search hiệu quả"""
        if not self.model:
            return [(sentence, 1.0)]
        
        self.model.eval()
        
        # Chuẩn bị input
        input_chars = list(sentence[:max_length])
        input_indices = [self.char_to_idx_input.get(c, 1) for c in input_chars]
        
        # Pad to max_length
        while len(input_indices) < max_length:
            input_indices.append(0)
        
        input_tensor = torch.tensor([input_indices]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)  # (1, seq_len, vocab_size)
            probs = F.softmax(outputs, dim=-1)
        
        # Beam search với heap
        beam = [BeamSearchNode("", 0.0, 0)]
        completed_sequences = []
        
        for pos in range(len(input_chars)):
            candidates = []
            
            for node in beam:
                # Lấy top-k characters cho position này
                pos_probs = probs[0, pos]
                top_k_probs, top_k_indices = torch.topk(pos_probs, min(beam_width * 2, pos_probs.size(0)))
                
                for prob, idx in zip(top_k_probs, top_k_indices):
                    char = self.idx_to_char_output.get(idx.item(), '<UNK>')
                    if char not in ['<PAD>', '<UNK>']:
                        new_score = node.score + torch.log(prob + 1e-8).item()  # Log probability
                        new_node = BeamSearchNode(node.sequence + char, new_score, pos + 1)
                        candidates.append(new_node)
            
            # Giữ top beam_width candidates
            candidates.sort(reverse=True)  # Sort by score descending
            beam = candidates[:beam_width]
        
        # Convert log probabilities to regular probabilities và normalize
        results = []
        for node in beam:
            prob = np.exp(node.score)
            results.append((node.sequence, prob))
        
        # Normalize probabilities
        total_prob = sum(prob for _, prob in results)
        if total_prob > 0:
            results = [(seq, prob / total_prob) for seq, prob in results]
        
        return results if results else [(sentence, 1.0)]
    
    def predict_word(self, word, top_k=5):
        """Dự đoán từ - sử dụng greedy decoding vì đã test hoạt động tốt"""
        if not self.model:
            return [(word, 1.0)]
        
        self.model.eval()
        
        # Convert to indices
        indices = [self.char_to_idx_input.get(c, 1) for c in word[:64]]
        while len(indices) < 64:
            indices.append(0)
        
        input_tensor = torch.tensor([indices]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=-1)
        
        # Method 1: Greedy decoding (đã test hoạt động tốt)
        greedy_result = ""
        greedy_score = 0.0
        
        for pos in range(len(word)):
            best_idx = torch.argmax(probs[0, pos]).item()
            best_char = self.idx_to_char_output.get(best_idx, '<UNK>')
            best_prob = probs[0, pos, best_idx].item()
            
            if best_char not in ['<PAD>', '<UNK>']:
                greedy_result += best_char
                greedy_score += np.log(best_prob + 1e-8)
        
        results = [(greedy_result, np.exp(greedy_score))]
        
        # Method 2: Generate alternatives bằng cách thay đổi từng position
        alternatives = set()
        
        for pos in range(len(word)):
            # Lấy top-3 chars cho position này
            top_3_probs, top_3_indices = torch.topk(probs[0, pos], min(3, probs.size(-1)))
            
            for prob, idx in zip(top_3_probs, top_3_indices):
                char = self.idx_to_char_output.get(idx.item(), '<UNK>')
                if char not in ['<PAD>', '<UNK>']:
                    # Tạo alternative bằng cách thay char tại position này
                    alternative = list(greedy_result)
                    if pos < len(alternative):
                        alternative[pos] = char
                        alt_word = ''.join(alternative)
                        
                        if alt_word != greedy_result and alt_word not in alternatives:
                            alternatives.add(alt_word)
                            
                            # Tính score cho alternative này
                            alt_score = 0.0
                            for p in range(len(word)):
                                if p < len(alt_word):
                                    char_idx = self.char_to_idx_output.get(alt_word[p], 1)
                                    if char_idx < probs.size(-1):
                                        char_prob = probs[0, p, char_idx].item()
                                        alt_score += np.log(char_prob + 1e-8)
                            
                            results.append((alt_word, np.exp(alt_score)))
        
        # Sắp xếp theo score và lấy top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]
        
        # Normalize probabilities
        if results:
            total_prob = sum(prob for _, prob in results)
            if total_prob > 0:
                results = [(seq, prob/total_prob) for seq, prob in results]
        
        return results if results else [(word, 1.0)]
    
    def restore_tones_optimized(self, text, max_results=5, use_sentence_level=True):
        """Khôi phục dấu tối ưu với tùy chọn sentence-level hoặc word-level"""
        if not text.strip():
            return [(text, 1.0)]
        
        # FORCE sử dụng word-level processing vì sentence-level beam search vẫn có bug
        # Sentence-level processing
        # if use_sentence_level and hasattr(self, '_trained_sentence_level') and self._trained_sentence_level:
        #     return self.predict_sentence_beam_search(text.lower(), beam_width=max_results)
        
        # Word-level processing (luôn dùng vì đã hoạt động tốt)
        words = re.findall(r'\w+', text.lower())
        if not words:
            return [(text, 1.0)]
        
        # Efficient beam search ở cấp độ câu thay vì tạo tất cả combinations
        sentence_beam = [("", 1.0, 0)]  # (current_sentence, score, word_index)
        
        for word_idx, word in enumerate(words):
            word_results = self.predict_word(word, top_k=min(max_results, 3))  # Giới hạn để tránh explosion
            
            new_beam = []
            for current_sentence, current_score, _ in sentence_beam:
                for predicted_word, word_score in word_results:
                    if current_sentence:
                        new_sentence = current_sentence + " " + predicted_word
                    else:
                        new_sentence = predicted_word
                    
                    new_score = current_score * word_score
                    new_beam.append((new_sentence, new_score, word_idx + 1))
            
            # Giữ top candidates
            new_beam.sort(key=lambda x: x[1], reverse=True)
            sentence_beam = new_beam[:max_results]
        
        # Normalize scores
        results = [(sentence, score) for sentence, score, _ in sentence_beam]
        if results:
            total_score = sum(score for _, score in results)
            if total_score > 0:
                results = [(sentence, score/total_score) for sentence, score in results]
        
        return results if results else [(text, 1.0)]
    
    def restore_tones(self, text, max_results=5):
        """Backward compatibility - wrapper cho restore_tones_optimized"""
        return self.restore_tones_optimized(text, max_results, use_sentence_level=True)
    
    def save_model(self, epoch=None, val_loss=None, is_best=False, mode="word-level"):
        """Lưu model với checkpoint theo epoch"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Get model configuration
        model_config = {
            'lightweight': self.lightweight,
            'mode': mode,
            'sentence_level': getattr(self, '_trained_sentence_level', mode == "sentence-level")
        }
        
        # Get actual model dimensions if model exists
        if self.model:
            embedding_dim = self.model.embedding.embedding_dim
            # Calculate hidden_dim from GRU
            gru_input_size = self.model.gru.input_size
            gru_hidden_size = self.model.gru.hidden_size
            model_config.update({
                'embedding_dim': embedding_dim,
                'hidden_dim': gru_hidden_size,
                'input_vocab_size': len(self.char_to_idx_input),
                'output_vocab_size': len(self.char_to_idx_output)
            })
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'char_to_idx_input': self.char_to_idx_input,
            'char_to_idx_output': self.char_to_idx_output,
            'idx_to_char_output': self.idx_to_char_output,
            'epoch': epoch,
            'val_loss': val_loss,
            'model_config': model_config  # Thêm thông tin cấu hình
        }
        
        # Lưu model chính (luôn update)
        main_path = os.path.join(self.model_dir, f'neural_tone_model_{mode}.pth')
        torch.save(checkpoint, main_path)
        
        # Lưu checkpoint theo epoch
        if epoch is not None:
            epoch_path = os.path.join(self.model_dir, f'checkpoint_epoch_{epoch:03d}_loss_{val_loss:.4f}_{mode}.pth')
            torch.save(checkpoint, epoch_path)
            print(f"Đã lưu checkpoint: {epoch_path}")
        
        # Lưu best model
        if is_best:
            best_path = os.path.join(self.model_dir, f'best_model_{mode}.pth')
            torch.save(checkpoint, best_path)
            print(f"Đã lưu best model: {best_path}")
    
    def cleanup_old_checkpoints(self, keep_last=3):
        """Xóa các checkpoint cũ, chỉ giữ lại một số checkpoint gần nhất"""
        if not os.path.exists(self.model_dir):
            return
            
        # Tìm tất cả checkpoint files
        checkpoint_files = []
        for filename in os.listdir(self.model_dir):
            if filename.startswith('checkpoint_epoch_') and filename.endswith('.pth'):
                filepath = os.path.join(self.model_dir, filename)
                checkpoint_files.append((filepath, filename))
        
        # Sort theo epoch (từ tên file)
        checkpoint_files.sort(key=lambda x: x[1])
        
        # Xóa các file cũ, chỉ giữ lại keep_last files
        if len(checkpoint_files) > keep_last:
            files_to_delete = checkpoint_files[:-keep_last]
            for filepath, filename in files_to_delete:
                try:
                    os.remove(filepath)
                    print(f"Đã xóa checkpoint cũ: {filename}")
                except Exception as e:
                    print(f"Không thể xóa {filename}: {e}")
    
    def load_model(self, use_best=True, mode="sentence-level"):
        """Load model với tùy chọn mode"""
        modes_to_try = [mode]
        if mode == "sentence-level":
            modes_to_try.append("word-level")  # Fallback
        elif mode == "word-level":
            modes_to_try.append("sentence-level")  # Fallback
        
        for current_mode in modes_to_try:
            print(f"Thử load model {current_mode}...")
            
            # Thử load best model trước
            if use_best:
                best_path = os.path.join(self.model_dir, f'best_model_{current_mode}.pth')
                if os.path.exists(best_path):
                    if self._load_checkpoint(best_path, current_mode):
                        return True
            
            # Thử load model chính
            main_path = os.path.join(self.model_dir, f'neural_tone_model_{current_mode}.pth')
            if os.path.exists(main_path):
                if self._load_checkpoint(main_path, current_mode):
                    return True
        
        # Thử load checkpoint gần nhất (bất kỳ mode nào)
        latest_checkpoint = self._find_latest_checkpoint()
        if latest_checkpoint:
            # Try to detect mode from filename, fallback to requested mode
            if "sentence-level" in latest_checkpoint:
                mode_from_file = "sentence-level"
            elif "word-level" in latest_checkpoint:
                mode_from_file = "word-level"
            else:
                mode_from_file = mode  # Use requested mode as fallback
            
            if self._load_checkpoint(latest_checkpoint, mode_from_file):
                return True
        
        return False
    
    def _load_checkpoint(self, model_path, mode="word-level"):
        """Load checkpoint từ path với thông tin mode"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.char_to_idx_input = checkpoint['char_to_idx_input']
            self.char_to_idx_output = checkpoint['char_to_idx_output']
            self.idx_to_char_output = checkpoint['idx_to_char_output']
            
            input_vocab_size = len(self.char_to_idx_input)
            output_vocab_size = len(self.char_to_idx_output)
            
            # Try to use saved model config first, fallback to auto-detection
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                detected_embedding_dim = model_config.get('embedding_dim', 128)
                detected_hidden_dim = model_config.get('hidden_dim', 256)
                is_lightweight_model = model_config.get('lightweight', False)
                saved_mode = model_config.get('mode', mode)
                self._trained_sentence_level = model_config.get('sentence_level', mode == "sentence-level")
                
                print(f"Using saved model config:")
                print(f"  Embedding dim: {detected_embedding_dim}")
                print(f"  Hidden dim: {detected_hidden_dim}")
                print(f"  Is lightweight: {is_lightweight_model}")
                print(f"  Original mode: {saved_mode}")
            else:
                # Fallback: Auto-detect model size từ checkpoint weights
                print("No saved config found, auto-detecting from weights...")
                model_state = checkpoint['model_state_dict']
                
                # Check embedding dimension từ checkpoint
                if 'embedding.weight' in model_state:
                    detected_embedding_dim = model_state['embedding.weight'].shape[1]
                else:
                    detected_embedding_dim = 128  # default
                
                # Check hidden dimension từ GRU weights
                if 'gru.weight_ih_l0' in model_state:
                    # GRU input weight shape: [3*hidden_size, embedding_dim] cho bidirectional
                    gru_input_size = model_state['gru.weight_ih_l0'].shape[0]
                    detected_hidden_dim = gru_input_size // 3  # 3 gates in GRU
                else:
                    detected_hidden_dim = 256  # default
                
                # Determine if model is lightweight based on detected dimensions
                is_lightweight_model = detected_embedding_dim <= 32 and detected_hidden_dim <= 64
                
                print(f"Auto-detected model size:")
                print(f"  Embedding dim: {detected_embedding_dim}")
                print(f"  Hidden dim: {detected_hidden_dim}")
                print(f"  Is lightweight: {is_lightweight_model}")
            
            # Override self.lightweight based on detected size
            original_lightweight = self.lightweight
            self.lightweight = is_lightweight_model
            
            self.model = ToneRestorationModel(input_vocab_size, output_vocab_size, 
                                            embedding_dim=detected_embedding_dim,
                                            hidden_dim=detected_hidden_dim,
                                            dropout_rate=0.3,
                                            lightweight=is_lightweight_model).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set training mode flag
            self._trained_sentence_level = (mode == "sentence-level")
            
            epoch = checkpoint.get('epoch', 'Unknown')
            val_loss = checkpoint.get('val_loss', 'Unknown')
            print(f"Neural model loaded from {model_path}")
            print(f"Mode: {mode}, Epoch: {epoch}, Val Loss: {val_loss}")
            print(f"Original lightweight setting: {original_lightweight}")
            print(f"Detected lightweight: {is_lightweight_model}")
            return True
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return False
    
    def _find_latest_checkpoint(self):
        """Tìm checkpoint gần nhất"""
        if not os.path.exists(self.model_dir):
            return None
            
        checkpoint_files = []
        for filename in os.listdir(self.model_dir):
            if filename.startswith('checkpoint_epoch_') and filename.endswith('.pth'):
                filepath = os.path.join(self.model_dir, filename)
                checkpoint_files.append((filepath, filename))
        
        if checkpoint_files:
            # Sort và lấy file gần nhất
            checkpoint_files.sort(key=lambda x: x[1])
            latest_path = checkpoint_files[-1][0]
            print(f"Tìm thấy checkpoint gần nhất: {checkpoint_files[-1][1]}")
            return latest_path
        
        return None

    def evaluate_model(self):
        """Evaluate model với test cases cố định"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        test_cases = [
            ("xin chao", "xin chào"),
            ("rat vui", "rất vui"), 
            ("cam on", "cảm ơn"),
            ("hom nay", "hôm nay"),
            ("ban co khoe khong", "bạn có khỏe không"),
            ("toi la sinh vien", "tôi là sinh viên"),
            ("chuc mung nam moi", "chúc mừng năm mới"),
            ("rat dep", "rất đẹp"),
            ("co gi moi", "có gì mới"),
            ("hen gap lai", "hẹn gặp lại"),
            ("anh ay thich hop", "anh ấy thích hợp"),  # Test ngữ cảnh câu
            ("toi di hoc", "tôi đi học"),
            ("mot ngay dep troi", "một ngày đẹp trời")
        ]
        
        total_correct = 0
        total_tests = len(test_cases)
        
        for input_text, expected in test_cases:
            results = self.restore_tones(input_text, max_results=1)
            predicted = results[0][0] if results else input_text
            score = results[0][1] if results else 0.0
            
            is_correct = predicted == expected
            if is_correct:
                total_correct += 1
                status = "DUNG"
            else:
                status = "SAI"
            
            print(f"{status} | Input: {input_text:25} | Expected: {expected:25} | Predicted: {predicted:25} | Score: {score:.3f}")
        
        accuracy = total_correct / total_tests * 100
        print(f"\nAccuracy: {total_correct}/{total_tests} = {accuracy:.1f}%")
        print("="*60)
        
        return accuracy
    
    def benchmark_performance(self, test_sentences=None):
        """Đo hiệu suất model"""
        if test_sentences is None:
            test_sentences = [
                "xin chao ban",
                "toi la sinh vien",
                "hom nay troi rat dep",
                "ban co khoe khong",
                "chuc mung nam moi"
            ]
        
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARK")
        print("="*50)
        
        total_time = 0
        num_tests = len(test_sentences)
        
        for sentence in test_sentences:
            start_time = time.time()
            results = self.restore_tones(sentence, max_results=3)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000  # ms
            total_time += processing_time
            
            print(f"Input: {sentence:20} | Time: {processing_time:6.2f}ms | Results: {len(results)}")
        
        avg_time = total_time / num_tests
        print(f"\nAverage processing time: {avg_time:.2f}ms")
        print(f"Total time: {total_time:.2f}ms")
        
        # Model size info
        if self.model:
            param_count = sum(p.numel() for p in self.model.parameters())
            model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming 32-bit floats
            print(f"Model parameters: {param_count:,}")
            print(f"Estimated model size: {model_size_mb:.2f} MB")
        
        print("="*50)
        return avg_time

def main():
    print("VIETNAMESE NEURAL TONE RESTORATION - ENHANCED BI-GRU MODEL")
    print("=" * 65)
    
    # Tùy chọn model
    import sys
    lightweight = "--lightweight" in sys.argv
    use_sentences = "--sentences" in sys.argv or "--sentence-level" in sys.argv
    train_only = "--train" in sys.argv
    
    mode_str = "sentence-level" if use_sentences else "word-level"
    weight_str = "lightweight" if lightweight else "standard"
    
    print(f"Mode: {mode_str}")
    print(f"Model: {weight_str}")
    
    neural_model = NeuralToneModel(lightweight=lightweight)
    
    # Thử load model
    model_loaded = neural_model.load_model(mode=mode_str)
    
    if not model_loaded or train_only:
        if model_loaded and train_only:
            print("Model đã tồn tại nhưng sẽ train lại...")
        else:
            print("Chưa có model, bắt đầu training...")
        
        # Training parameters
        epochs = 20 if lightweight else 50
        batch_size = 256 if lightweight else 128
        
        neural_model.train(
            dict_path="data/Viet74K_clean.txt",
            corpus_path="data/cleaned_comments.txt",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.001,
            weight_decay=1e-4,
            use_sentences=use_sentences
        )
        
        # Set training mode sau khi train
        neural_model._trained_sentence_level = use_sentences
    
    # Evaluate model
    neural_model.evaluate_model()
    
    # Performance benchmark
    neural_model.benchmark_performance()
    
    # Interactive test
    print("\nINTERACTIVE TEST:")
    print("Nhập 'quit' để thoát")
    print("Nhập '--help' để xem các lệnh đặc biệt")
    
    while True:
        text = input("\nNhập văn bản: ").strip()
        if text.lower() == 'quit':
            break
        elif text == '--help':
            print("Các lệnh đặc biệt:")
            print("  --benchmark : Chạy performance benchmark")
            print("  --eval      : Chạy evaluation lại")
            print("  --stats     : Hiển thị thông tin model")
            continue
        elif text == '--benchmark':
            neural_model.benchmark_performance()
            continue
        elif text == '--eval':
            neural_model.evaluate_model()
            continue
        elif text == '--stats':
            if neural_model.model:
                param_count = sum(p.numel() for p in neural_model.model.parameters())
                model_size_mb = param_count * 4 / (1024 * 1024)
                print(f"Model parameters: {param_count:,}")
                print(f"Model size: {model_size_mb:.2f} MB")
                print(f"Mode: {mode_str}")
                print(f"Lightweight: {lightweight}")
                print(f"Device: {neural_model.device}")
            continue
            
        if text:
            start_time = time.time()
            results = neural_model.restore_tones(text, max_results=5)
            end_time = time.time()
            
            print("Kết quả:")
            for i, (restored, score) in enumerate(results, 1):
                print(f"  {i}. {restored} (score: {score:.4f})")
            
            print(f"Time: {(end_time - start_time)*1000:.2f}ms")

if __name__ == "__main__":
    main() 