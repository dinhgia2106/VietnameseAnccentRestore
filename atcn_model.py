#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention-based Temporal Convolutional Network (A-TCN) for Vietnamese Accent Restoration

This module implements a hybrid architecture combining Temporal Convolutional Networks (TCN)
with Multi-Head Self-Attention for Vietnamese accent restoration tasks.
"""

import math
from typing import Dict, List, Tuple, Optional
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from config import MODEL_CONFIG, VIETNAMESE_CHARS


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sinusoidal functions.
    
    This encoding allows the model to understand the position of tokens
    in the sequence without relying on recurrent connections.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalConvBlock(nn.Module):
    """
    Temporal Convolutional Block with dilated convolutions.
    
    Uses dilated convolutions to capture long-range dependencies
    while maintaining computational efficiency.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=padding)
        
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.residual = (nn.Conv1d(in_channels, out_channels, 1) 
                        if in_channels != out_channels 
                        else nn.Identity())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal conv block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor with same shape as input
        """
        x_input = x
        
        # Convert to conv format: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # First convolution block
        out = self.conv1(x)
        out = out.transpose(1, 2)  # Back to (batch, seq_len, features)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution block
        out = out.transpose(1, 2)
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        
        # Residual connection
        residual = self.residual(x_input.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out + residual)
        
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Allows the model to attend to different positions in the sequence
    and capture complex dependencies between characters.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor with same shape as input
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(out)


class ATCN(nn.Module):
    """
    Attention-based Temporal Convolutional Network.
    
    Main model that combines TCN layers for local feature extraction
    with attention layers for global context understanding.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 8,
                 num_tcn_layers: int = 6, num_attention_layers: int = 3,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layers
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # TCN layers with exponentially increasing dilation
        self.tcn_layers = nn.ModuleList()
        for i in range(num_tcn_layers):
            dilation = 2 ** i
            self.tcn_layers.append(
                TemporalConvBlock(d_model, d_model, kernel_size, dilation, dropout)
            )
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList()
        self.attention_norms = nn.ModuleList()
        for _ in range(num_attention_layers):
            self.attention_layers.append(
                MultiHeadSelfAttention(d_model, num_heads, dropout)
            )
            self.attention_norms.append(nn.LayerNorm(d_model))
        
        # Output layers
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the A-TCN model.
        
        Args:
            x: Input tensor of shape (batch, seq_len) with token IDs
            mask: Optional attention mask
            
        Returns:
            Output logits of shape (batch, seq_len, vocab_size)
        """
        # Embedding with scaled initialization
        x = self.char_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Apply TCN layers for local feature extraction
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)
        
        # Apply attention layers for global context
        for attention_layer, norm in zip(self.attention_layers, self.attention_norms):
            residual = x
            x = attention_layer(x, mask)
            x = norm(x + residual)
            x = self.dropout(x)
        
        # Final output projection
        output = self.output_projection(x)
        
        return output


class VietnameseTokenizer:
    """
    Tokenizer for Vietnamese text that handles both accented and non-accented characters.
    
    Includes special tokens for sequence processing and comprehensive Vietnamese character set.
    """
    
    def __init__(self):
        # Build character vocabulary from config
        self.chars = self._build_vocabulary()
        
        # Create mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.chars)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Special token IDs
        self.unk_id = self.char_to_id['UNK']
        self.pad_id = self.char_to_id['PAD']
        self.bos_id = self.char_to_id['BOS']
        self.eos_id = self.char_to_id['EOS']
    
    def _build_vocabulary(self) -> List[str]:
        """Build comprehensive Vietnamese vocabulary from config."""
        vocabulary = ['UNK', 'PAD', 'BOS', 'EOS', ' ']  # Special tokens
        
        # Add character sets from config
        vocabulary.extend(sorted(VIETNAMESE_CHARS['base_chars']))
        vocabulary.extend(sorted(VIETNAMESE_CHARS['accented_chars']))
        vocabulary.extend(sorted(VIETNAMESE_CHARS['digits']))
        vocabulary.extend(sorted(VIETNAMESE_CHARS['punctuation']))
        
        return vocabulary
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        text = text.lower().strip()
        tokens = [self.bos_id]
        
        for char in text:
            token_id = self.char_to_id.get(char, self.unk_id)
            tokens.append(token_id)
        
        tokens.append(self.eos_id)
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        chars = []
        for token_id in token_ids:
            if token_id in [self.bos_id, self.eos_id, self.pad_id]:
                continue
            
            char = self.id_to_char.get(token_id, 'UNK')
            if char != 'UNK':
                chars.append(char)
        
        return ''.join(chars).strip()
    
    @staticmethod
    def remove_accents(text: str) -> str:
        """Remove Vietnamese accents using Unicode normalization."""
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        return text.lower()


class ContextRanker(nn.Module):
    """
    Context-aware ranking module for suggestion scoring.
    
    Uses bidirectional LSTM to understand context and rank suggestions
    based on contextual appropriateness.
    """
    
    def __init__(self, d_model: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        self.context_encoder = nn.LSTM(
            d_model, hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.scoring_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, context_emb: torch.Tensor, 
                suggestion_embs: torch.Tensor) -> torch.Tensor:
        """
        Rank suggestions based on context.
        
        Args:
            context_emb: Context embeddings (batch, seq_len, d_model)
            suggestion_embs: Suggestion embeddings (batch, num_suggestions, seq_len, d_model)
            
        Returns:
            Ranking scores (batch, num_suggestions)
        """
        batch_size, num_suggestions = suggestion_embs.shape[:2]
        
        # Encode context
        context_encoded, _ = self.context_encoder(context_emb)
        context_pooled = context_encoded.mean(dim=1)  # Pool over sequence length
        
        # Score each suggestion
        scores = []
        for i in range(num_suggestions):
            suggestion_emb = suggestion_embs[:, i, :, :]
            suggestion_encoded, _ = self.context_encoder(suggestion_emb)
            suggestion_pooled = suggestion_encoded.mean(dim=1)
            
            # Combine context and suggestion
            combined = context_pooled + suggestion_pooled
            score = self.scoring_network(combined)
            scores.append(score)
        
        scores = torch.stack(scores, dim=1)  # (batch, num_suggestions, 1)
        return scores.squeeze(-1)  # (batch, num_suggestions)


def create_model(vocab_size: Optional[int] = None, 
                device: str = 'cpu') -> Tuple[ATCN, VietnameseTokenizer, ContextRanker]:
    """
    Create and initialize the complete model stack.
    
    Args:
        vocab_size: Vocabulary size (if None, will be inferred from tokenizer)
        device: Device to place models on
        
    Returns:
        Tuple of (ATCN model, Tokenizer, ContextRanker)
    """
    # Initialize tokenizer
    tokenizer = VietnameseTokenizer()
    
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    
    # Create models with config parameters
    atcn_config = MODEL_CONFIG['atcn']
    atcn_model = ATCN(
        vocab_size=vocab_size,
        d_model=atcn_config['d_model'],
        num_heads=atcn_config['num_heads'],
        num_tcn_layers=atcn_config['num_tcn_layers'],
        num_attention_layers=atcn_config['num_attention_layers'],
        kernel_size=atcn_config['kernel_size'],
        dropout=atcn_config['dropout']
    )
    
    context_ranker = ContextRanker(
        d_model=atcn_config['d_model'],
        hidden_dim=MODEL_CONFIG['context_ranker']['hidden_dim']
    )
    
    # Move to device
    atcn_model = atcn_model.to(device)
    context_ranker = context_ranker.to(device)
    
    return atcn_model, tokenizer, context_ranker


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Demo usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Creating model on device: {device}")
    
    atcn_model, tokenizer, context_ranker = create_model(device=device)
    
    print(f"ATCN parameters: {count_parameters(atcn_model):,}")
    print(f"Context Ranker parameters: {count_parameters(context_ranker):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Test encoding/decoding
    test_text = "may bay"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: '{test_text}' -> Encoded: {encoded} -> Decoded: '{decoded}'") 