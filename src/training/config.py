"""
Training Configuration for A-TCN
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


@dataclass
class TrainingConfig:
    """Configuration for A-TCN training"""
    
    # Model parameters
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 6
    kernel_size: int = 3
    dropout: float = 0.1
    max_dilation: int = 32
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 50
    max_length: int = 256
    clip_grad_norm: float = 1.0
    
    # Data parameters
    corpus_dir: str = "corpus_splitted"
    vocab_path: str = "vietnamese_char_vocab.json"
    train_split: float = 0.9
    max_files: Optional[int] = None
    max_samples_per_file: int = 10000
    num_workers: int = 0
    
    # Optimization parameters
    lr_scheduler: str = "plateau"  # "plateau", "cosine", "linear"
    lr_decay_factor: float = 0.5
    lr_patience: int = 5
    warmup_steps: int = 1000
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    
    # Checkpointing
    output_dir: str = "outputs"
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 1000
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Inference
    temperature: float = 1.0
    top_k: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'model': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'kernel_size': self.kernel_size,
                'dropout': self.dropout,
                'max_dilation': self.max_dilation
            },
            'training': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'max_length': self.max_length,
                'clip_grad_norm': self.clip_grad_norm
            },
            'data': {
                'corpus_dir': self.corpus_dir,
                'vocab_path': self.vocab_path,
                'train_split': self.train_split,
                'max_files': self.max_files,
                'max_samples_per_file': self.max_samples_per_file,
                'num_workers': self.num_workers
            },
            'optimization': {
                'lr_scheduler': self.lr_scheduler,
                'lr_decay_factor': self.lr_decay_factor,
                'lr_patience': self.lr_patience,
                'warmup_steps': self.warmup_steps
            },
            'early_stopping': {
                'patience': self.early_stopping_patience,
                'delta': self.early_stopping_delta
            },
            'checkpointing': {
                'output_dir': self.output_dir,
                'save_every_n_epochs': self.save_every_n_epochs,
                'keep_last_n_checkpoints': self.keep_last_n_checkpoints
            },
            'logging': {
                'log_every_n_steps': self.log_every_n_steps,
                'eval_every_n_steps': self.eval_every_n_steps
            },
            'device': self.device,
            'inference': {
                'temperature': self.temperature,
                'top_k': self.top_k
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        # Flatten nested dict
        flat_config = {}
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_config[key] = value
            else:
                flat_config[section] = values
        
        return cls(**flat_config)
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.kernel_size > 0, "kernel_size must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.max_length > 0, "max_length must be positive"
        assert 0 <= self.train_split <= 1, "train_split must be between 0 and 1"
        assert self.lr_scheduler in ["plateau", "cosine", "linear"], f"Unknown lr_scheduler: {self.lr_scheduler}"
        
        print("Configuration validation passed")


# Default configurations for different scenarios
DEFAULT_CONFIG = TrainingConfig()

SMALL_CONFIG = TrainingConfig(
    embedding_dim=64,
    hidden_dim=128,
    num_layers=4,
    batch_size=16,
    max_files=10,
    max_samples_per_file=1000
)

LARGE_CONFIG = TrainingConfig(
    embedding_dim=256,
    hidden_dim=512,
    num_layers=8,
    batch_size=64,
    max_dilation=64,
    num_epochs=100
) 