import os
import gc
import psutil

def clear_system_resources():
    # Force Python Garbage Collection (System RAM)
    gc.collect()
    
    #  Kill "Zombie" Gradio or background processes
 
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        print(f"Terminating background process: {child.pid}")
        child.terminate()
    
    # Clear the Linux OS Buffer Cache (Kaggle runs on Linux)
    # This is a 'soft' way to tell the OS to free up inactive RAM
    os.system('sync') 
    
    print("RAM and CPU threads cleared.")

clear_system_resources()