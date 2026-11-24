from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_RESIZE_SIZE = 256
_CROP_SIZE = 224

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR


class InvalidImageError(ValueError):
    pass

def preprocess_image(raw: bytes) -> np.ndarray:

    if not raw:
        raise InvalidImageError("empty file")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(str(exc)) from exc


    if img.mode != "RGB":
        img = img.convert("RGB")


    img = _resize_short_side(img, _RESIZE_SIZE)

    img = _center_crop(img, _CROP_SIZE)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)                     # CHW
    arr = (arr - _MEAN[:, None, None]) / _STD[:, None, None]
    return arr[None, ...].astype(np.float32, copy=False)  # NCHW


def _resize_short_side(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w <= h:
        new_w, new_h = size, int(round(h * size / w))
    else:
        new_h, new_w = size, int(round(w * size / h))
    return img.resize((new_w, new_h), _BILINEAR)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = int(round((w - size) / 2.0))
    top = int(round((h - size) / 2.0))
    return img.crop((left, top, left + size, top + size))