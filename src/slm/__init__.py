"""slm — a small, highly capable language model you can train and run locally."""
from .config import ModelConfig, TrainConfig, preset
from .model import KVCache, Transformer
from .tokenizer import SLMTokenizer

__all__ = ["ModelConfig", "TrainConfig", "preset", "Transformer", "KVCache", "SLMTokenizer"]
__version__ = "0.1.0"
