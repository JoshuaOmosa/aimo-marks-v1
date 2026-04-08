import torch
import gc

def clear_gpu_memory():
    
    vars_to_kill = ['model', 'base_model', 'trainer', 'tokenizer']
    for var in vars_to_kill:
        if var in globals():
            del globals()[var]
    
    # 2. Force Garbage Collection
    gc.collect()
    
    # 3. Clear the CUDA Cache (The hardware level)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize() # Wait for all kernels to finish
        
    print("GPU VRAM cleared and synchronized.")

clear_gpu_memory()