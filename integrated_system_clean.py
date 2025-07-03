#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Vietnamese Accent Restoration System

This module combines N-gram and A-TCN approaches to provide comprehensive
accent restoration with contextual ranking.
"""

import os
import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

from config import get_device, MODEL_CONFIG


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Data class for prediction results."""
    text: str
    confidence: float
    source: str
    context_score: Optional[float] = None


class IntegratedAccentSystem:
    """
    Integrated Vietnamese Accent Restoration System.
    
    Combines multiple approaches:
    1. N-gram based suggestions for statistical patterns
    2. A-TCN model for deep learning predictions
    3. Context ranking for intelligent suggestion ordering
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the integrated system.
        
        Args:
            device: Computing device ('cpu', 'cuda', etc.). Auto-detected if None.
        """
        self.device = device or get_device()
        logger.info(f"Initializing integrated system on device: {self.device}")
        
        # Component systems (loaded lazily)
        self.ngram_system = None
        self.atcn_model = None
        self.atcn_tokenizer = None
        self.context_ranker = None
        
        self._is_initialized = False
    
    def initialize_components(self):
        """Initialize all system components."""
        if self._is_initialized:
            return
        
        logger.info("Loading system components...")
        
        try:
            self._load_ngram_system()
            self._load_atcn_model()
            self._is_initialized = True
            logger.info("All components loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _load_ngram_system(self):
        """Load the N-gram restoration system."""
        try:
            from vietnamese_accent_restore import VietnameseAccentRestore
            self.ngram_system = VietnameseAccentRestore(
                max_ngram=MODEL_CONFIG['max_ngram']
            )
            logger.info("N-gram system loaded successfully")
            
        except ImportError as e:
            logger.warning(f"N-gram system not available: {e}")
        except Exception as e:
            logger.error(f"Error loading N-gram system: {e}")
    
    def _load_atcn_model(self):
        """Load the A-TCN model if available."""
        try:
            from atcn_model import create_model
            
            self.atcn_model, self.atcn_tokenizer, self.context_ranker = create_model(
                device=self.device
            )
            
            # Try to load pre-trained weights if available
            model_paths = [
                os.path.join("models", "best_model.pth"),
                os.path.join("models", "demo", "best_model.pth")
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        checkpoint = torch.load(model_path, map_location=self.device)
                        self.atcn_model.load_state_dict(checkpoint['model_state_dict'])
                        logger.info(f"Pre-trained A-TCN model loaded from: {model_path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load model from {model_path}: {e}")
            
            if not model_loaded:
                logger.info("A-TCN model created (no pre-trained weights found)")
                
        except ImportError as e:
            logger.warning(f"A-TCN model not available: {e}")
        except Exception as e:
            logger.error(f"Error loading A-TCN model: {e}")
    
    def predict_ngram(self, input_text: str, max_suggestions: int = 5) -> List[Prediction]:
        """Get predictions from N-gram system."""
        if not self.ngram_system:
            return []
        
        try:
            suggestions = self.ngram_system.find_suggestions(
                input_text, max_suggestions=max_suggestions
            )
            
            return [
                Prediction(text=text, confidence=conf, source='ngram')
                for text, conf in suggestions
            ]
            
        except Exception as e:
            logger.error(f"N-gram prediction error: {e}")
            return []
    
    def predict_atcn(self, input_text: str, max_suggestions: int = 5) -> List[Prediction]:
        """Get predictions from A-TCN model."""
        if not self.atcn_model or not self.atcn_tokenizer:
            return []
        
        try:
            # Create a temporary trainer to use prediction methods
            from train_atcn import ATCNTrainer
            
            trainer = ATCNTrainer(self.atcn_model, self.atcn_tokenizer, self.device)
            
            # Get multiple predictions using enhanced method
            atcn_predictions = trainer.predict_text_multiple(
                input_text, 
                max_suggestions=max_suggestions,
                use_beam_search=True
            )
            
            # Convert to Prediction objects
            predictions = []
            for text, confidence in atcn_predictions:
                predictions.append(
                    Prediction(
                        text=text, 
                        confidence=confidence * 100,  # Convert to percentage
                        source='atcn'
                    )
                )
            
            return predictions
            
        except Exception as e:
            logger.error(f"A-TCN prediction error: {e}")
            return []
    
    def rank_with_context(self, predictions: List[Prediction], 
                         context: str) -> List[Prediction]:
        """Apply context-based ranking to predictions."""
        if not self.context_ranker or not predictions:
            return predictions
        
        try:
            # This would need more sophisticated implementation
            # For now, just return original predictions
            logger.debug("Context ranking not fully implemented yet")
            return predictions
            
        except Exception as e:
            logger.error(f"Context ranking error: {e}")
            return predictions
    
    def predict_hybrid(self, input_text: str, 
                      max_suggestions: int = 5,
                      context: Optional[str] = None) -> List[Prediction]:
        """
        Get hybrid predictions combining all available approaches.
        
        Args:
            input_text: Input text without accents
            max_suggestions: Maximum number of suggestions to return
            context: Optional surrounding context for better ranking
            
        Returns:
            List of predictions sorted by confidence
        """
        if not self._is_initialized:
            self.initialize_components()
        
        all_predictions = []
        
        # Collect predictions from all sources
        ngram_predictions = self.predict_ngram(input_text, max_suggestions)
        atcn_predictions = self.predict_atcn(input_text, max_suggestions)
        
        all_predictions.extend(ngram_predictions)
        all_predictions.extend(atcn_predictions)
        
        # Remove duplicates and merge confidence scores
        merged_predictions = self._merge_predictions(all_predictions)
        
        # Apply context ranking if available
        if context:
            merged_predictions = self.rank_with_context(merged_predictions, context)
        
        # Sort by confidence and return top suggestions
        merged_predictions.sort(key=lambda x: x.confidence, reverse=True)
        return merged_predictions[:max_suggestions]
    
    def _merge_predictions(self, predictions: List[Prediction]) -> List[Prediction]:
        """Merge duplicate predictions and combine confidence scores."""
        prediction_map: Dict[str, Prediction] = {}
        
        for pred in predictions:
            if pred.text in prediction_map:
                # Combine confidence scores (weighted average)
                existing = prediction_map[pred.text]
                combined_confidence = (existing.confidence + pred.confidence) / 2
                
                # Update with higher confidence and multi-source info
                prediction_map[pred.text] = Prediction(
                    text=pred.text,
                    confidence=combined_confidence,
                    source=f"{existing.source}+{pred.source}"
                )
            else:
                prediction_map[pred.text] = pred
        
        return list(prediction_map.values())
    
    def interactive_demo(self):
        """Run interactive demonstration of the integrated system."""
        print("HE THONG TICH HOP KHOI PHUC DAU TIENG VIET")
        print("=" * 60)
        print("Ket hop A-TCN + N-gram + Context Ranking")
        print("Nhap van ban khong dau de nhan goi y (go 'quit' de thoat)")
        print()
        
        while True:
            try:
                user_input = input("Nhap van ban: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Tam biet!")
                    break
                
                if not user_input:
                    continue
                
                # Get predictions
                predictions = self.predict_hybrid(user_input, max_suggestions=10)
                
                if predictions:
                    print(f"\nGoi y cho '{user_input}':")
                    for i, pred in enumerate(predictions, 1):
                        print(f"  {i}. {pred.text} "
                              f"(tin cay: {pred.confidence:.1f}, nguon: {pred.source})")
                else:
                    print(f"Khong tim thay goi y cho '{user_input}'")
                
                print()
                
            except KeyboardInterrupt:
                print("\nTam biet!")
                break
            except Exception as e:
                logger.error(f"Demo error: {e}")
                print(f"Loi: {e}")
    
    def batch_predict(self, texts: List[str], 
                     max_suggestions: int = 3) -> Dict[str, List[Prediction]]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of input texts
            max_suggestions: Maximum suggestions per text
            
        Returns:
            Dictionary mapping input texts to their predictions
        """
        results = {}
        
        for text in texts:
            try:
                predictions = self.predict_hybrid(text, max_suggestions)
                results[text] = predictions
            except Exception as e:
                logger.error(f"Error processing '{text}': {e}")
                results[text] = []
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status information about loaded components."""
        return {
            "device": self.device,
            "ngram_available": self.ngram_system is not None,
            "atcn_available": self.atcn_model is not None,
            "context_ranker_available": self.context_ranker is not None,
            "initialized": self._is_initialized
        }


def test_system():
    """Test the integrated system with common phrases."""
    system = IntegratedAccentSystem()
    system.initialize_components()
    
    test_cases = [
        "may bay",
        "cam on",
        "hoc sinh", 
        "co giao",
        "truong hoc"
    ]
    
    print("TESTING INTEGRATED SYSTEM")
    print("=" * 40)
    print(f"System status: {system.get_system_status()}")
    print()
    
    for test_case in test_cases:
        predictions = system.predict_hybrid(test_case, max_suggestions=3)
        print(f"{test_case} -> {[p.text for p in predictions]}")


def main():
    """Main function to run the integrated system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Vietnamese Accent Restoration')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    parser.add_argument('--device', type=str, help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = IntegratedAccentSystem(device=args.device)
    
    if args.test:
        test_system()
    else:
        system.interactive_demo()


if __name__ == "__main__":
    main() 