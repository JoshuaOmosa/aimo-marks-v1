import gc

import torch

DEFAULT_NAMES = ("model", "base_model", "trainer", "tokenizer")


def clear_gpu_memory(namespace=None, names=DEFAULT_NAMES):
    """Free GPU memory held by a training or inference session.

    Pass the caller's namespace (in a notebook or script: ``globals()``) to
    drop references to large objects such as the model and trainer. Without
    that, Python keeps them alive and ``empty_cache()`` can't release them.
    """
    if namespace is not None:
        for name in names:
            namespace.pop(name, None)

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # wait for pending kernels before reporting
        print(f"GPU memory cleared. Still allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print("Python objects released (no CUDA device present).")


if __name__ == "__main__":
    clear_gpu_memory()
