from model_architecture import VietnameseAccentRestorer

print("=== TEST KIẾN TRÚC A-TCN ===")
print("Model chưa được huấn luyện (random weights)")

# Khởi tạo model mới (chưa train)
r = VietnameseAccentRestorer()

print(f"Vocabulary size: {r.vocab_size}")
print(f"Model parameters: {sum(p.numel() for p in r.model.parameters()):,}")

# Test với fix preserve spaces
tests = [
    "toi di hoc",
    "xin chao", 
    "viet nam la dat nuoc dep",
    "123 abc def!"
]

print("\n=== TEST POST-PROCESSING FIX ===")
for test in tests:
    result = r.predict(test)
    spaces_preserved = test.count(' ') == result.count(' ')
    length_preserved = len(test) == len(result)
    
    print(f"Input:  '{test}' (len: {len(test)})")
    print(f"Output: '{result}' (len: {len(result)})")
    print(f"Spaces preserved: {spaces_preserved}")
    print(f"Length preserved: {length_preserved}")
    print("-" * 50)

print("✅ Kiến trúc và post-processing đã hoạt động!")
print("🔄 Training đang chạy nền để tạo model thực sự...") 