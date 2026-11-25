"""Taşınabilir Seed-VC HuggingFace Model İndirici (Yalnızca ./_modeller/ içine yazar)."""

from __future__ import annotations

from huggingface_hub import hf_hub_download

from core.portable_env import MODELS_DIR, bootstrap_portable_environment


def get_seed_vc_cache_dir() -> str:
    """Seed-VC ağırlıklarının indirileceği taşınabilir ./_modeller/huggingface/hub dizinini döndürür."""
    bootstrap_portable_environment()
    hub_dir = MODELS_DIR / "huggingface" / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    return str(hub_dir)


def load_custom_model_from_hf(
    repo_id: str,
    model_filename: str = "pytorch_model.bin",
    config_filename: str | None = None,
):
    """Belirtilen Seed-VC model ve yapılandırma dosyasını yalnızca ./_modeller/ içine indirir."""
    cache_dir = get_seed_vc_cache_dir()
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        cache_dir=cache_dir,
    )
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        cache_dir=cache_dir,
    )
    return model_path, config_path
