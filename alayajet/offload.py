import mlx.core as mx
import shutil
from pathlib import Path

class OffloadManager:
    """
    Manages the storage of KV cache layers on disk.
    Strictly handles file I/O and memory clearing logic.
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.reset_storage()
        self.layer_counter = 0

    def reset_storage(self):
        """Clean up and recreate the cache directory."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def reset_counter(self):
        """Reset layer counter. Should be called at start of generation."""
        self.layer_counter = 0

    def get_file_path(self, layer_idx: int) -> Path:
        return self.cache_dir / f"layer_{layer_idx}.safetensors"

    def save_layer(self, layer_idx: int, keys: mx.array, values: mx.array, offset: int):
        """
        Save keys/values to disk and return None to indicate memory should be cleared.
        """
        mx.eval(keys, values)
        data = {"keys": keys, "values": values}
        metadata = {"offset": str(offset)}
        mx.save_safetensors(str(self.get_file_path(layer_idx)), data, metadata)
        return None, None

    def load_layer(self, layer_idx: int):
        """
        Load keys/values from disk. Returns (keys, values, offset).
        Returns (None, None, 0) if file not found.
        """
        path = self.get_file_path(layer_idx)
        if not path.exists():
            return None, None, 0
        arrays, metadata = mx.load(str(path), return_metadata=True)
        return arrays["keys"], arrays["values"], int(metadata["offset"])
