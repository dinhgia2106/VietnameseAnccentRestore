"""
A-TCN (Acausal Temporal Convolutional Network) for Vietnamese Accent Restoration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import numpy as np


class DilatedCausalConv1d(nn.Module):
    """
    Dilated Causal 1D Convolution for TCN
    - Causal: output at time t only depends on inputs up to time t
    - Dilated: increases receptive field exponentially
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        # Causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, seq_len]
        Returns:
            [batch_size, channels, seq_len]
        """
        # Apply conv1d
        out = self.conv1d(x)
        
        # Remove future information (causal)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        return self.dropout(out)


class AcausalDilatedConv1d(nn.Module):
    """
    Acausal (Bidirectional) Dilated Convolution
    Uses both past and future context - key innovation of A-TCN
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation // 2
        
        # Bidirectional convolution (sees past + future)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            dilation=dilation,
            padding=self.padding
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, seq_len]
        Returns:
            [batch_size, channels, seq_len]
        """
        out = self.conv1d(x)
        return self.dropout(out)


class TCNResidualBlock(nn.Module):
    """
    TCN Residual Block with:
    - Two dilated convolutions
    - Layer normalization
    - Residual connection
    - Gated activation
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1, acausal: bool = True):
        super().__init__()
        
        self.acausal = acausal
        conv_layer = AcausalDilatedConv1d if acausal else DilatedCausalConv1d
        
        # First conv layer
        self.conv1 = conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout
        )
        
        # Second conv layer
        self.conv2 = conv_layer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Gated activation
        self.activation = nn.GELU()
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, seq_len]
        Returns:
            [batch_size, channels, seq_len] 
        """
        # Store residual
        residual = x
        
        # First conv + norm + activation
        out = self.conv1(x)
        out = out.transpose(1, 2)  # [batch, seq_len, channels] for LayerNorm
        out = self.norm1(out)
        out = self.activation(out)
        out = out.transpose(1, 2)  # Back to [batch, channels, seq_len]
        
        # Second conv + norm
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
            
        out = out + residual
        return self.activation(out)


class ATCN(nn.Module):
    """
    Acausal Temporal Convolutional Network for Vietnamese Accent Restoration
    
    Key features:
    - Character-level input/output
    - Bidirectional context (acausal)
    - Exponential dilation for large receptive field
    - Residual connections for deep networks
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 hidden_dim: int = 256, num_layers: int = 8,
                 kernel_size: int = 3, dropout: float = 0.1,
                 max_dilation: int = 256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Input projection
        self.input_projection = nn.Conv1d(embedding_dim, hidden_dim, 1)
        
        # TCN layers with exponential dilation
        self.tcn_layers = nn.ModuleList()
        dilations = [min(2**i, max_dilation) for i in range(num_layers)]
        
        for i, dilation in enumerate(dilations):
            layer = TCNResidualBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                acausal=True
            )
            self.tcn_layers.append(layer)
        
        # Output projection
        self.output_projection = nn.Conv1d(hidden_dim, vocab_size, 1)
        
        # Initialize weights
        self._init_weights()
        
        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field(kernel_size, dilations)
        
        print(f"A-TCN Model created:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Dilations: {dilations}")
        print(f"  Receptive field: {self.receptive_field}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def _calculate_receptive_field(self, kernel_size: int, dilations: List[int]) -> int:
        """Calculate receptive field of the network"""
        receptive_field = 1
        for dilation in dilations:
            receptive_field += (kernel_size - 1) * dilation
        return receptive_field
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Character embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Transpose for conv1d: [batch_size, embedding_dim, seq_len]
        x = embedded.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)  # [batch_size, hidden_dim, seq_len]
        
        # TCN layers
        for layer in self.tcn_layers:
            x = layer(x)
        
        # Output projection
        logits = self.output_projection(x)  # [batch_size, vocab_size, seq_len]
        
        # Transpose back: [batch_size, seq_len, vocab_size]
        logits = logits.transpose(1, 2)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Set padded positions to large negative value
            mask = attention_mask.unsqueeze(-1).expand_as(logits)
            logits = logits.masked_fill(~mask.bool(), -1e9)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: Optional[int] = None,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate text (for inference)
        
        Args:
            input_ids: [batch_size, seq_len]
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            generated_ids: [batch_size, seq_len]
        """
        self.eval()
        
        if max_length is None:
            max_length = input_ids.shape[1]
        
        with torch.no_grad():
            # Get model predictions
            logits = self.forward(input_ids)  # [batch_size, seq_len, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            generated_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
            generated_ids = generated_ids.view(logits.shape[:2])
        
        return generated_ids
    
    def generate_with_constraints(self, input_ids: torch.Tensor, 
                                char_processor, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate text with Vietnamese accent constraints
        
        Args:
            input_ids: [batch_size, seq_len] 
            char_processor: VietnameseCharProcessor instance
            temperature: Sampling temperature
        
        Returns:
            generated_ids: [batch_size, seq_len]
        """
        self.eval()
        
        # Define Vietnamese vowel mapping
        vowel_mappings = {
            'a': ['a', 'à', 'á', 'ả', 'ã', 'ạ'],
            'ă': ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ'],
            'â': ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ'],
            'e': ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ'],
            'ê': ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ'],
            'i': ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị'],
            'o': ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ'],
            'ô': ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ'],
            'ơ': ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ'],
            'u': ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ'],
            'ư': ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự'],
            'y': ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ'],
            # Uppercase
            'A': ['A', 'À', 'Á', 'Ả', 'Ã', 'Ạ'],
            'Ă': ['Ă', 'Ằ', 'Ắ', 'Ẳ', 'Ẵ', 'Ặ'],
            'Â': ['Â', 'Ầ', 'Ấ', 'Ẩ', 'Ẫ', 'Ậ'],
            'E': ['E', 'È', 'É', 'Ẻ', 'Ẽ', 'Ẹ'],
            'Ê': ['Ê', 'Ề', 'Ế', 'Ể', 'Ễ', 'Ệ'],
            'I': ['I', 'Ì', 'Í', 'Ỉ', 'Ĩ', 'Ị'],
            'O': ['O', 'Ò', 'Ó', 'Ỏ', 'Õ', 'Ọ'],
            'Ô': ['Ô', 'Ồ', 'Ố', 'Ổ', 'Ỗ', 'Ộ'],
            'Ơ': ['Ơ', 'Ờ', 'Ớ', 'Ở', 'Ỡ', 'Ợ'],
            'U': ['U', 'Ù', 'Ú', 'Ủ', 'Ũ', 'Ụ'],
            'Ư': ['Ư', 'Ừ', 'Ứ', 'Ử', 'Ữ', 'Ự'],
            'Y': ['Y', 'Ỳ', 'Ý', 'Ỷ', 'Ỹ', 'Ỵ'],
        }
        
        with torch.no_grad():
            # Get model predictions
            logits = self.forward(input_ids)  # [batch_size, seq_len, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            batch_size, seq_len = input_ids.shape
            generated_ids = torch.zeros_like(input_ids)
            
            # Process each position
            for b in range(batch_size):
                for pos in range(seq_len):
                    input_char_idx = input_ids[b, pos].item()
                    
                    # Skip special tokens
                    if input_char_idx in [0, 1, 2, 3]:  # PAD, UNK, BOS, EOS
                        generated_ids[b, pos] = input_char_idx
                        continue
                    
                    # Get input character
                    input_char = char_processor.idx_to_char.get(input_char_idx, '')
                    
                    # Check if this character can be modified
                    if input_char in vowel_mappings:
                        # Get valid target characters for this vowel
                        valid_chars = vowel_mappings[input_char]
                        valid_indices = [char_processor.char_to_idx.get(c, input_char_idx) 
                                       for c in valid_chars 
                                       if c in char_processor.char_to_idx]
                        
                        # Create mask for valid indices
                        pos_logits = logits[b, pos].clone()
                        mask = torch.full_like(pos_logits, -float('inf'))
                        for valid_idx in valid_indices:
                            mask[valid_idx] = pos_logits[valid_idx]
                        
                        # Sample from valid characters only
                        probs = F.softmax(mask, dim=-1)
                        generated_ids[b, pos] = torch.multinomial(probs, 1)
                    else:
                        # Keep consonants, numbers, punctuation unchanged
                        generated_ids[b, pos] = input_char_idx
        
        return generated_ids 