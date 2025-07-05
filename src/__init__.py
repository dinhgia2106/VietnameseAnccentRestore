"""
Vietnamese Accent Restoration using A-TCN
"""

from .models import ATCN, ATCNTrainer
from .data import VietnameseCharProcessor, VietnameseCharDataset, create_data_loaders
from .training import ATCNTrainingPipeline, TrainingConfig

__version__ = "1.0.0"
__author__ = "Vietnamese Accent Restoration Team"
__email__ = "contact@example.com"

__all__ = [
    'ATCN',
    'ATCNTrainer', 
    'VietnameseCharProcessor',
    'VietnameseCharDataset',
    'create_data_loaders',
    'ATCNTrainingPipeline',
    'TrainingConfig'
] 