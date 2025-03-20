import torch
import sys

def check_gpu():
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"可用的GPU數量: {device_count}")
        
        for i in range(device_count):
            print(f"\nGPU #{i}:")
            print(f"名稱: {torch.cuda.get_device_name(i)}")
            print(f"顯存總量: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")
            print(f"計算能力: {torch.cuda.get_device_capability(i)}")
            
        # 測試簡單操作
        print("\n執行GPU張量操作測試:")
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            z = torch.matmul(x, y)
            end_event.record()
            
            torch.cuda.synchronize()
            print(f"矩陣乘法計算時間: {start_event.elapsed_time(end_event):.3f} ms")
            print("GPU操作成功!")
        except Exception as e:
            print(f"GPU操作失敗: {e}")
    else:
        print("未檢測到可用的GPU。訓練將使用CPU，速度會顯著降低。")
        print("如果您有NVIDIA GPU，請確保已安裝正確的CUDA和cuDNN版本。")

if __name__ == "__main__":
    check_gpu() 