import torch
import torch.nn as nn
import torch.nn.functional as F

class ACausalTCN(nn.Module):
    """
    Acausal Temporal Convolutional Network cho phục hồi dấu tiếng Việt
    Dựa trên nghiên cứu của Alqahtani et al. (EMNLP 2019)
    """
    
    def __init__(self, vocab_size, output_size, embedding_dim=128, hidden_dim=256, num_layers=4, kernel_size=3, dropout=0.1):
        super(ACausalTCN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # TCN layers với dilated convolution
        self.tcn_layers = nn.ModuleList()
        
        # Lớp đầu tiên
        self.tcn_layers.append(
            TCNBlock(embedding_dim, hidden_dim, kernel_size, dilation=1, dropout=dropout)
        )
        
        # Các lớp tiếp theo với dilation tăng dần
        for i in range(1, num_layers):
            dilation = 2 ** i
            self.tcn_layers.append(
                TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation=dilation, dropout=dropout)
            )
        
        # Layer classification cuối
        self.classifier = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # Qua các TCN layers
        out = embedded
        for tcn_layer in self.tcn_layers:
            out = tcn_layer(out)
        
        out = out.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        out = self.dropout(out)
        
        # Classification cho từng ký tự
        logits = self.classifier(out)  # (batch_size, seq_len, output_size)
        
        return logits

class TCNBlock(nn.Module):
    """
    Temporal Convolutional Block với dilated convolution và residual connection
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
        
        # Padding để đảm bảo output length = input length (causal padding)
        # Với acausal, padding ở cả 2 bên
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=self.padding // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding // 2)
        
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = out.transpose(1, 2)  # (batch_size, seq_len, channels)
        out = self.norm1(out)
        out = out.transpose(1, 2)  # (batch_size, channels, seq_len)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second conv block  
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
            
        # Crop để đảm bảo kích thước khớp (do padding)
        if out.size(2) != residual.size(2):
            diff = out.size(2) - residual.size(2)
            if diff > 0:
                out = out[:, :, diff//2:-(diff-diff//2)] if diff > 1 else out[:, :, diff//2:]
            
        out = out + residual
        return F.relu(out)

class VietnameseAccentRestorer:
    """
    Wrapper class cho mô hình phục hồi dấu tiếng Việt
    """
    
    def __init__(self, model_path=None):
        # Bảng ký tự tiếng Việt
        self.no_accent_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-()[]{}\"'/"
        self.accent_chars = "aăâeêioôơuưyAĂÂEÊIOÔƠUƯYáắấéếíóốớúứýÁẮẤÉẾÍÓỐỚÚỨÝàằầèềìòồờùừỳÀẰẦÈỀÌÒỒỜÙỪỲảẳẩẻểỉỏổởủửỷẢẲẨẺỂỈỎỔỞỦỬỶãẵẫẽễĩõỗỡũữỹÃẴẪẼỄĨÕỖỠŨỮỸạặậẹệịọộợụựỵẠẶẬẸỆỊỌỘỢỤỰỴđĐ"
        
        # Tạo vocabulary
        all_chars = set(self.no_accent_chars + self.accent_chars)
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        self.vocab_size = len(self.char_to_idx)
        self.output_size = self.vocab_size
        
        # Khởi tạo mô hình
        self.model = ACausalTCN(
            vocab_size=self.vocab_size,
            output_size=self.output_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=4,
            kernel_size=3,
            dropout=0.1
        )
        
        if model_path:
            self.load_model(model_path)
    
    def encode_text(self, text):
        """Chuyển text thành sequence of indices"""
        return [self.char_to_idx.get(char, 0) for char in text]
    
    def decode_text(self, indices):
        """Chuyển sequence of indices về text"""
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])
    
    def predict(self, text):
        """Dự đoán text có dấu từ text không dấu"""
        self.model.eval()
        
        # Encode input
        input_ids = self.encode_text(text)
        input_tensor = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
        
        # Check device và move tensor đến cùng device với model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            predictions = torch.argmax(logits, dim=-1)
        
        # Decode output
        predicted_indices = predictions.squeeze().tolist()
        predicted_text = self.decode_text(predicted_indices)
        
        # Post-processing: Giữ nguyên khoảng trắng và ký tự đặc biệt
        corrected_text = self.preserve_non_alphabetic(text, predicted_text)
        
        return corrected_text
    
    def preserve_non_alphabetic(self, original, predicted):
        """
        Giữ nguyên khoảng trắng và ký tự không phải chữ cái từ text gốc
        """
        if len(original) != len(predicted):
            return predicted  # Fallback nếu length không khớp
        
        result = []
        for i, (orig_char, pred_char) in enumerate(zip(original, predicted)):
            if orig_char.isalpha():
                # Nếu là chữ cái, dùng kết quả dự đoán
                result.append(pred_char)
            else:
                # Nếu không phải chữ cái (space, số, dấu câu), giữ nguyên
                result.append(orig_char)
        
        return ''.join(result)
    
    def save_model(self, path):
        """Lưu mô hình"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'output_size': self.output_size
        }, path)
    
    def load_model(self, path):
        """Tải mô hình"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        self.vocab_size = checkpoint['vocab_size']
        self.output_size = checkpoint['output_size']
    
    def export_to_onnx(self, onnx_path, max_seq_len=512):
        """Xuất mô hình sang ONNX format"""
        self.model.eval()
        
        # Kiểm tra device của model
        device = next(self.model.parameters()).device
        
        # Tạo dummy input trên cùng device với model
        dummy_input = torch.randint(0, self.vocab_size, (1, max_seq_len)).to(device)
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {1: 'sequence'},
                'output': {1: 'sequence'}
            }
        )
        print(f"Đã xuất mô hình ONNX tại: {onnx_path}")

if __name__ == "__main__":
    # Test mô hình
    restorer = VietnameseAccentRestorer()
    
    # In thông tin mô hình
    total_params = sum(p.numel() for p in restorer.model.parameters())
    print(f"Tổng số tham số: {total_params:,}")
    print(f"Kích thước vocabulary: {restorer.vocab_size}")
    
    # Test forward pass
    test_text = "toi di hoc"
    encoded = restorer.encode_text(test_text)
    print(f"Text: {test_text}")
    print(f"Encoded: {encoded}") 