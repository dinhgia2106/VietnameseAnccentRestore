from model_architecture import VietnameseAccentRestorer
import torch

def debug_model():
    print("=== DEBUG MÔ HÌNH PHỤC HỒI DẤU ===")
    
    # Khởi tạo mô hình
    restorer = VietnameseAccentRestorer('models/best_model.pth')
    
    print(f"Vocabulary size: {restorer.vocab_size}")
    print(f"Sample vocab: {list(restorer.char_to_idx.keys())[:30]}")
    
    # Test encoding/decoding
    test_texts = ["toi di hoc", "xin chao", "a b c"]
    
    for text in test_texts:
        print(f"\n--- Test: '{text}' ---")
        
        # Encode
        encoded = restorer.encode_text(text)
        print(f"Original: '{text}' (len: {len(text)})")
        print(f"Encoded:  {encoded}")
        
        # Decode back
        decoded = restorer.decode_text(encoded)
        print(f"Decoded:  '{decoded}' (len: {len(decoded)})")
        print(f"Match original: {text == decoded}")
        
        # Model prediction
        restorer.model.eval()
        input_tensor = torch.tensor(encoded).unsqueeze(0)
        
        # Check device
        device = next(restorer.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            logits = restorer.model(input_tensor)
            predictions = torch.argmax(logits, dim=-1)
        
        predicted_indices = predictions.squeeze().tolist()
        predicted_text = restorer.decode_text(predicted_indices)
        
        print(f"Predicted indices: {predicted_indices}")
        print(f"Predicted text: '{predicted_text}' (len: {len(predicted_text)})")
        print(f"Length preserved: {len(text) == len(predicted_text)}")
        
        # Character-by-character comparison
        print("Char-by-char:")
        max_len = max(len(text), len(predicted_text))
        for i in range(max_len):
            orig_char = text[i] if i < len(text) else '?'
            pred_char = predicted_text[i] if i < len(predicted_text) else '?'
            print(f"  {i:2d}: '{orig_char}' -> '{pred_char}'")

if __name__ == "__main__":
    debug_model() 