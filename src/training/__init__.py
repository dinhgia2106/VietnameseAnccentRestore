"""
Vietnamese Accent Restoration - Training Package
"""

try:
    from .pipeline import ATCNTrainingPipeline
    from .config import TrainingConfig, SMALL_CONFIG, LARGE_CONFIG, DEFAULT_CONFIG
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from pipeline import ATCNTrainingPipeline
    from config import TrainingConfig, SMALL_CONFIG, LARGE_CONFIG, DEFAULT_CONFIG

__all__ = ['ATCNTrainingPipeline', 'TrainingConfig', 'SMALL_CONFIG', 'LARGE_CONFIG', 'DEFAULT_CONFIG'] 