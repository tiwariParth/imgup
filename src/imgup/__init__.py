import os

# cuDNN's workspace probing logs scary-but-harmless allocator OOM warnings at large tiles.
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
