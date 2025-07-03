#!/usr/bin/env python3
"""
DEMO SCRIPT - DUY NHẤT
Vietnamese Accent Restoration Multi-Suggestion Demo
"""

import os
from vietnamese_accent_restore import VietnameseAccentRestore

def demo_multi_suggestions():
    """Demo multi-suggestion capabilities."""
    
    print("🇻🇳 VIETNAMESE ACCENT RESTORATION - MULTI-SUGGESTION DEMO")
    print("=" * 70)
    
    # Initialize N-gram system
    try:
        model = VietnameseAccentRestore()
        print("✅ N-gram model loaded successfully!")
        print()
        
    except Exception as e:
        print(f"❌ Error loading N-gram model: {e}")
        return
    
    # Test cases
    test_cases = [
        ("toi", "Single word - multiple meanings"),
        ("may bay", "Two words - different contexts"),
        ("cam on", "Common phrase"),
        ("toi di hoc", "Complete sentence"),
        ("hom nay troi dep", "Weather description"),
        ("ban co khoe khong", "Question")
    ]
    
    print("🔍 TESTING MULTI-SUGGESTIONS:")
    print("-" * 50)
    
    for input_text, description in test_cases:
        print(f"\n📝 Input: '{input_text}' ({description})")
        
        try:
            # Get multiple suggestions
            suggestions = model.find_suggestions(input_text, max_suggestions=5)
            
            if suggestions:
                print("   Suggestions:")
                for i, (text, score) in enumerate(suggestions, 1):
                    print(f"     {i}. '{text}' (score: {score:.1f})")
            else:
                print("   ❌ No suggestions found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("🎯 MULTI-SUGGESTION FEATURES DEMONSTRATED:")
    print("✅ Single word variations (toi → tôi, tới, tối, tội)")  
    print("✅ Context-aware suggestions (may bay → máy bay, mây bay)")
    print("✅ Frequency-based ranking")
    print("✅ Configurable max_suggestions parameter")
    
    # A-TCN Demo (if available)
    atcn_model_path = "models/best_model.pth"
    print(f"\n🤖 A-TCN MODEL STATUS:")
    
    if os.path.exists(atcn_model_path):
        print(f"✅ A-TCN model found: {atcn_model_path}")
        print("💡 Run integrated demo for A-TCN + N-gram combination")
    else:
        print("⏳ A-TCN model not trained yet")
        print("💡 Run: python train.py")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Train A-TCN model: python train.py")
    print("2. Test integrated system with both N-gram + A-TCN")
    print("3. Compare single vs multi-suggestion performance")


if __name__ == "__main__":
    demo_multi_suggestions() 