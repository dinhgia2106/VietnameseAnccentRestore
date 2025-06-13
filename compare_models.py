import torch
import time
from model_architecture import VietnameseAccentRestorer

def compare_models():
    """
    So sánh performance và khả năng của model cũ vs model mới
    """
    print("So sánh Model Cũ vs Model Enhanced")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        "xin chao ban",
        "toi la nguoi viet nam",
        "hom nay troi dep qua",
        "chung ta cung nhau hoc tap",
        "viet nam la dat nuoc xinh dep",
        "tieng viet co nhieu dau sac",
        "machine learning rat thu vi",
        "deep learning va natural language processing"
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")
    print()
    
    # Khởi tạo models
    print("Khởi tạo models...")
    old_model = VietnameseAccentRestorer(use_enhanced_model=False)
    enhanced_model = VietnameseAccentRestorer(use_enhanced_model=True)
    
    old_model.model.to(device)
    enhanced_model.model.to(device)
    
    # So sánh số lượng parameters
    old_params = sum(p.numel() for p in old_model.model.parameters())
    enhanced_params = sum(p.numel() for p in enhanced_model.model.parameters())
    
    print(f"Model cũ: {old_params:,} parameters")
    print(f"Model enhanced: {enhanced_params:,} parameters")
    print(f"Tăng {enhanced_params/old_params:.1f}x parameters")
    print()
    
    # Test predictions
    print("Test predictions:")
    print("-" * 50)
    
    old_model.model.eval()
    enhanced_model.model.eval()
    
    total_old_time = 0
    total_enhanced_time = 0
    
    for i, test_text in enumerate(test_cases):
        print(f"Test {i+1}: {test_text}")
        
        # Test old model
        start_time = time.time()
        old_prediction = old_model.predict(test_text)
        old_time = time.time() - start_time
        total_old_time += old_time
        
        # Test enhanced model
        start_time = time.time()
        enhanced_prediction = enhanced_model.predict(test_text)
        enhanced_time = time.time() - start_time
        total_enhanced_time += enhanced_time
        
        print(f"  Input:    {test_text}")
        print(f"  Old:      {old_prediction} ({old_time:.3f}s)")
        print(f"  Enhanced: {enhanced_prediction} ({enhanced_time:.3f}s)")
        print()
    
    # Thống kê thời gian
    print("Thống kê thời gian:")
    print(f"Model cũ trung bình: {total_old_time/len(test_cases):.3f}s")
    print(f"Model enhanced trung bình: {total_enhanced_time/len(test_cases):.3f}s")
    
    if total_enhanced_time > total_old_time:
        print(f"Enhanced model chậm hơn {total_enhanced_time/total_old_time:.1f}x")
    else:
        print(f"Enhanced model nhanh hơn {total_old_time/total_enhanced_time:.1f}x")

def benchmark_memory():
    """
    Benchmark memory usage
    """
    print("\nBenchmark Memory Usage:")
    print("-" * 30)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test old model
        old_model = VietnameseAccentRestorer(use_enhanced_model=False)
        old_model.model.cuda()
        
        test_input = torch.randint(0, 100, (8, 256)).cuda()
        _ = old_model.model(test_input)
        
        old_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        print(f"Model cũ: {old_memory:.1f} MB")
        
        del old_model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test enhanced model
        enhanced_model = VietnameseAccentRestorer(use_enhanced_model=True)
        enhanced_model.model.cuda()
        
        _ = enhanced_model.model(test_input)
        
        enhanced_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        print(f"Model enhanced: {enhanced_memory:.1f} MB")
        print(f"Tăng {enhanced_memory/old_memory:.1f}x memory")
        
        del enhanced_model
        torch.cuda.empty_cache()
    else:
        print("CUDA không có sẵn, bỏ qua memory benchmark")

if __name__ == "__main__":
    compare_models()
    benchmark_memory() 