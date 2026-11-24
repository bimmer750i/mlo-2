from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from config import settings


def detect_gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        name = out.stdout.strip().splitlines()[0].strip()
        if name:
            return name
    except (subprocess.SubprocessError, FileNotFoundError, IndexError):
        pass
    return settings.gpu_name or "unknown"


def load_labels() -> list[str]:
    path = Path(__file__).parent / "imagenet_classes.txt"
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def softmax_top5(logits: np.ndarray) -> list[tuple[int, float]]:
    x = logits - logits.max()
    probs = np.exp(x)
    probs /= probs.sum()
    top5 = np.argsort(probs)[-5:][::-1]
    return [(int(i), float(probs[int(i)])) for i in top5]