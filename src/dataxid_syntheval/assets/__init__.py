from __future__ import annotations

from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent


def load_embedder():
    """Load the bundled potion-base-8M static embedding model."""
    from model2vec import StaticModel

    model_path = _MODULE_DIR / "embedders" / "potion-base-8M"
    return StaticModel.from_pretrained(model_path)
