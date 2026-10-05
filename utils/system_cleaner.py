import gc
import os

import psutil


def clear_system_resources(terminate_children=True):
    """Release RAM and stop leftover child processes (e.g. a stuck Gradio server).

    Useful on Kaggle/Colab between runs. Only children of the current process
    are touched, so nothing else on the machine is affected.
    """
    gc.collect()

    if terminate_children:
        for child in psutil.Process().children(recursive=True):
            print(f"Terminating background process: {child.pid}")
            child.terminate()

    # Flush filesystem buffers so the OS can reclaim page cache (Linux/macOS only).
    if hasattr(os, "sync"):
        os.sync()

    print("RAM released and child processes stopped.")


if __name__ == "__main__":
    clear_system_resources()
