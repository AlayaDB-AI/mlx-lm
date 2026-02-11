import os

# Set HF endpoint before importing anything that might load huggingface_hub.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from .server import main


if __name__ == "__main__":
    main()
