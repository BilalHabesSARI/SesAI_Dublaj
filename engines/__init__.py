"""Hibrit İşlem Motorları (Colab Pro Bulut + Yerel NVIDIA CUDA GPU + ElevenLabs API)."""

from engines.base import SpeechEngine, TranscriptionResult
from engines.colab_engine import ColabCloudEngine
from engines.colab_notebook_generator import generate_colab_notebook
from engines.elevenlabs_engine import ElevenLabsEngine
from engines.local_gpu_engine import LocalGPUEngine

__all__ = [
    "SpeechEngine",
    "TranscriptionResult",
    "ColabCloudEngine",
    "LocalGPUEngine",
    "ElevenLabsEngine",
    "generate_colab_notebook",
]
