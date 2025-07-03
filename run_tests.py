#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Vietnamese Accent Restoration System (Clean Version)

This script runs comprehensive tests for all system components.
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logging, progress_bar
from config import get_config_summary

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


class TestRunner:
    """Test runner for all system components."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.test_cases = [
            "may bay",
            "cam on", 
            "hoc sinh",
            "co giao",
            "truong hoc",
            "tieu hoc",
            "nha truong",
            "giao vien"
        ]
    
    def test_utils(self) -> bool:
        """Test utility functions."""
        logger.info("Testing utility functions...")
        
        try:
            from utils import (
                normalize_vietnamese_text,
                remove_vietnamese_accents, 
                is_valid_vietnamese_text,
                clean_text_for_processing,
                split_into_sentences
            )
            
            # Test normalization
            test_text = "Xin  chào   tôi là    hệ thống"
            normalized = normalize_vietnamese_text(test_text)
            assert "  " not in normalized, "Normalization failed"
            
            # Test accent removal
            accented = "máy bay"
            no_accent = remove_vietnamese_accents(accented)
            assert no_accent == "may bay", f"Accent removal failed: {no_accent}"
            
            # Test Vietnamese text validation
            assert is_valid_vietnamese_text("xin chào"), "Vietnamese validation failed"
            assert not is_valid_vietnamese_text("hello world"), "English should not validate as Vietnamese"
            
            # Test sentence splitting
            long_text = "Đây là câu đầu tiên. Đây là câu thứ hai! Đây là câu thứ ba?"
            sentences = split_into_sentences(long_text, max_length=50)
            assert len(sentences) >= 3, "Sentence splitting failed"
            
            logger.info("✓ Utility functions tests passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Utility functions test failed: {e}")
            return False
    
    def test_config(self) -> bool:
        """Test configuration system."""
        logger.info("Testing configuration system...")
        
        try:
            from config import MODEL_CONFIG, TRAINING_CONFIG, get_device
            
            # Test config structure
            assert 'max_ngram' in MODEL_CONFIG, "Missing max_ngram config"
            assert 'atcn' in MODEL_CONFIG, "Missing atcn config"
            assert 'batch_size' in TRAINING_CONFIG, "Missing batch_size config"
            
            # Test device detection
            device = get_device()
            assert device in ['cpu', 'cuda'], f"Invalid device: {device}"
            
            # Test config summary
            summary = get_config_summary()
            assert "Vietnamese Accent Restoration" in summary, "Config summary failed"
            
            logger.info("✓ Configuration tests passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Configuration test failed: {e}")
            return False
    
    def test_ngram_system(self) -> bool:
        """Test N-gram accent restoration system."""
        logger.info("Testing N-gram system...")
        
        try:
            from vietnamese_accent_restore import VietnameseAccentRestore
            
            # Initialize with limited n-grams for faster testing
            restore = VietnameseAccentRestore(max_ngram=5)
            
            # Test basic functionality
            suggestions = restore.find_suggestions("may bay", max_suggestions=3)
            assert len(suggestions) > 0, "No suggestions returned"
            
            # Check suggestion format
            for suggestion, confidence in suggestions:
                assert isinstance(suggestion, str), "Invalid suggestion type"
                assert isinstance(confidence, (int, float)), "Invalid confidence type"
                assert confidence > 0, "Invalid confidence value"
            
            # Test multiple test cases
            success_count = 0
            for test_case in self.test_cases[:5]:  # Test subset for speed
                suggestions = restore.find_suggestions(test_case, max_suggestions=3)
                if suggestions:
                    success_count += 1
                    logger.debug(f"'{test_case}' -> {[s[0] for s in suggestions[:2]]}")
            
            assert success_count >= 3, f"Too few successful predictions: {success_count}/5"
            
            logger.info(f"✓ N-gram system tests passed ({success_count}/5 test cases)")
            self.results['ngram_success_rate'] = success_count / 5
            return True
            
        except Exception as e:
            logger.error(f"✗ N-gram system test failed: {e}")
            return False
    
    def test_atcn_model(self) -> bool:
        """Test A-TCN model components."""
        logger.info("Testing A-TCN model...")
        
        try:
            from atcn_model import create_model, VietnameseTokenizer
            
            # Test tokenizer
            tokenizer = VietnameseTokenizer()
            
            # Test encoding/decoding
            test_text = "may bay"
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded)
            
            assert isinstance(encoded, list), "Encoding should return list"
            assert len(encoded) > 0, "Encoded should not be empty"
            assert isinstance(decoded, str), "Decoding should return string"
            
            # Test model creation
            device = 'cpu'  # Use CPU for testing
            atcn_model, tokenizer, context_ranker = create_model(device=device)
            
            assert atcn_model is not None, "ATCN model creation failed"
            assert tokenizer is not None, "Tokenizer creation failed"
            assert context_ranker is not None, "Context ranker creation failed"
            
            # Test model inference (basic)
            import torch
            sample_input = torch.tensor([encoded], device=device)
            
            with torch.no_grad():
                output = atcn_model(sample_input)
                assert output.shape[0] == 1, "Batch dimension incorrect"
                assert output.shape[1] == len(encoded), "Sequence length incorrect"
                assert output.shape[2] == tokenizer.vocab_size, "Vocab dimension incorrect"
            
            logger.info("✓ A-TCN model tests passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ A-TCN model test failed: {e}")
            return False
    
    def test_integrated_system(self) -> bool:
        """Test integrated system."""
        logger.info("Testing integrated system...")
        
        try:
            from integrated_system_clean import IntegratedAccentSystem, Prediction
            
            # Initialize system
            system = IntegratedAccentSystem(device='cpu')
            
            # Test component loading
            system.initialize_components()
            status = system.get_system_status()
            
            assert status['device'] == 'cpu', "Device mismatch"
            assert 'ngram_available' in status, "Missing ngram status"
            assert status['initialized'], "System not initialized"
            
            # Test predictions
            predictions = system.predict_hybrid("may bay", max_suggestions=3)
            
            assert isinstance(predictions, list), "Predictions should be list"
            
            if predictions:  # If we have predictions
                for pred in predictions:
                    assert isinstance(pred, Prediction), "Invalid prediction type"
                    assert hasattr(pred, 'text'), "Missing text attribute"
                    assert hasattr(pred, 'confidence'), "Missing confidence attribute"
                    assert hasattr(pred, 'source'), "Missing source attribute"
            
            # Test batch processing
            batch_results = system.batch_predict(self.test_cases[:3], max_suggestions=2)
            assert isinstance(batch_results, dict), "Batch results should be dict"
            assert len(batch_results) == 3, "Batch size mismatch"
            
            logger.info("✓ Integrated system tests passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Integrated system test failed: {e}")
            return False
    
    def benchmark_performance(self) -> Dict[str, float]:
        """Benchmark system performance."""
        logger.info("Running performance benchmarks...")
        
        benchmarks = {}
        
        try:
            # N-gram performance
            from vietnamese_accent_restore import VietnameseAccentRestore
            restore = VietnameseAccentRestore(max_ngram=5)
            
            start_time = time.time()
            for test_case in self.test_cases:
                restore.find_suggestions(test_case, max_suggestions=3)
            ngram_time = (time.time() - start_time) / len(self.test_cases)
            benchmarks['ngram_avg_time_ms'] = ngram_time * 1000
            
            # Integrated system performance
            from integrated_system_clean import IntegratedAccentSystem
            system = IntegratedAccentSystem(device='cpu')
            system.initialize_components()
            
            start_time = time.time()
            for test_case in self.test_cases:
                system.predict_hybrid(test_case, max_suggestions=3)
            integrated_time = (time.time() - start_time) / len(self.test_cases)
            benchmarks['integrated_avg_time_ms'] = integrated_time * 1000
            
            logger.info(f"✓ Benchmarks completed")
            logger.info(f"  N-gram avg time: {benchmarks['ngram_avg_time_ms']:.2f}ms")
            logger.info(f"  Integrated avg time: {benchmarks['integrated_avg_time_ms']:.2f}ms")
            
        except Exception as e:
            logger.error(f"✗ Benchmark failed: {e}")
        
        return benchmarks
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        logger.info("Starting comprehensive test suite...")
        logger.info("=" * 60)
        
        # Print configuration
        print(get_config_summary())
        
        test_results = {
            'utils': False,
            'config': False,
            'ngram': False,
            'atcn': False,
            'integrated': False,
            'benchmarks': {}
        }
        
        # Run tests
        tests = [
            ('utils', self.test_utils),
            ('config', self.test_config),
            ('ngram', self.test_ngram_system),
            ('atcn', self.test_atcn_model),
            ('integrated', self.test_integrated_system)
        ]
        
        for i, (name, test_func) in enumerate(tests):
            print(f"\n{progress_bar(i, len(tests), f'Running {name} tests')}")
            test_results[name] = test_func()
        
        # Run benchmarks
        print(f"\n{progress_bar(len(tests), len(tests), 'Running benchmarks')}")
        test_results['benchmarks'] = self.benchmark_performance()
        
        # Summary
        print(f"\n{progress_bar(len(tests), len(tests), 'Tests completed')}")
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for result in test_results.values() if result is True)
        total = len([k for k in test_results.keys() if k != 'benchmarks'])
        
        for name, result in test_results.items():
            if name != 'benchmarks':
                status = "PASS" if result else "FAIL"
                logger.info(f"{name.upper():<12}: {status}")
        
        logger.info(f"\nOVERALL: {passed}/{total} tests passed")
        
        if 'ngram_success_rate' in self.results:
            logger.info(f"N-gram success rate: {self.results['ngram_success_rate']*100:.1f}%")
        
        return test_results


def main():
    """Main function to run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Vietnamese Accent Restoration System')
    parser.add_argument('--component', type=str, help='Test specific component (utils/config/ngram/atcn/integrated)')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks only')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logging(log_level="DEBUG")
    
    runner = TestRunner()
    
    if args.benchmark:
        runner.benchmark_performance()
    elif args.component:
        test_func = getattr(runner, f'test_{args.component}', None)
        if test_func:
            test_func()
        else:
            logger.error(f"Unknown component: {args.component}")
    else:
        results = runner.run_all_tests()
        
        # Exit with error code if any tests failed
        failed = any(result is False for key, result in results.items() if key != 'benchmarks')
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main() 