from __future__ import annotations

import numpy as np
import httpx

from config import settings

_INPUT_NAME = "INPUT__0"
_OUTPUT_NAME = "OUTPUT__0"

limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
_client = httpx.AsyncClient(
    base_url=settings.triton_url,
    timeout=httpx.Timeout(settings.http_timeout),
    limits=limits
)

class TritonError(RuntimeError):
    pass


async def close() -> None:
    await _client.aclose()


async def is_ready() -> bool:

    try:
        resp = await _client.get("/v2/health/ready")
    except httpx.HTTPError:
        return False
    return resp.status_code == 200


async def infer(image_nchw: np.ndarray) -> np.ndarray:

    if image_nchw.shape != (1, 3, 224, 224):
        raise TritonError(f"unexpected input shape: {image_nchw.shape}")

    payload = {
        "inputs": [
            {
                "name": _INPUT_NAME,
                "shape": list(image_nchw.shape),
                "datatype": "FP32",
                # tolist() — обязательно: JSON не умеет ndarray
                "data": image_nchw.reshape(-1).tolist(),
            }
        ],
        "outputs": [{"name": _OUTPUT_NAME}],
    }

    try:
        resp = await _client.post(
            f"/v2/models/{settings.cv_model}/infer",
            json=payload,
        )
    except httpx.HTTPError as exc:
        raise TritonError(f"request failed: {exc}") from exc

    if resp.status_code != 200:

        raise TritonError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        body = resp.json()
        out = next(o for o in body["outputs"] if o["name"] == _OUTPUT_NAME)
        data = np.asarray(out["data"], dtype=np.float32)
    except (KeyError, StopIteration, ValueError) as exc:
        raise TritonError(f"malformed response: {exc}") from exc

    expected = int(np.prod(out.get("shape", [1000])))
    if data.size != expected:
        raise TritonError(f"output size mismatch: {data.size} != {expected}")

    return data.reshape(-1)