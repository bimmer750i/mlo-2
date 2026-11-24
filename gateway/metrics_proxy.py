from __future__ import annotations

import httpx
from fastapi import HTTPException, Response

from config import settings

_client = httpx.AsyncClient(timeout=httpx.Timeout(settings.http_timeout))

class MetricsUnavailable(RuntimeError):
    pass

async def close() -> None:
    await _client.aclose()

async def _fetch(url: str) -> str:
    try:
        resp = await _client.get(url)
    except httpx.HTTPError as exc:
        raise MetricsUnavailable(f"request failed: {exc}") from exc
    if resp.status_code != 200:
        raise MetricsUnavailable(f"HTTP {resp.status_code}")
    return resp.text


def _plain(text: str) -> Response:
    return Response(content=text, media_type="text/plain; charset=utf-8")

async def triton_metrics() -> Response:
    try:
        text = await _fetch(f"{settings.triton_metrics_url}/metrics")
    except MetricsUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return _plain(text)

async def vllm_metrics() -> Response:
    try:
        text = await _fetch(f"{settings.vllm_url}/metrics")
    except MetricsUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return _plain(text)

async def node_metrics() -> Response:
    try:
        text = await _fetch(f"{settings.node_exporter_url}/metrics")
    except MetricsUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return _plain(text)