from neural_model import NeuralToneModel

# Test model
model = NeuralToneModel(lightweight=True)
model.load_model()

# Test cases
test_cases = [
    "xin chao",
    "ban co khoe khong", 
    "toi la sinh vien",
    "cam on ban"
]

print("QUICK TEST RESULTS:")
print("=" * 40)

for text in test_cases:
    results = model.restore_tones(text, max_results=1)
    predicted = results[0][0] if results else text
    score = results[0][1] if results else 0.0
    
    print(f"'{text}' -> '{predicted}' ({score:.3f})") 