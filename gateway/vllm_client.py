from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from config import settings

_client = httpx.AsyncClient(
    base_url=settings.vllm_url,
    timeout=httpx.Timeout(settings.http_timeout),
)


class VLLMError(RuntimeError):
    pass


class VLLMRejected(VLLMError):
    pass


async def close() -> None:
    await _client.aclose()


async def is_ready() -> tuple[bool, list[str]]:
    try:
        resp = await _client.get("/v1/models")
    except httpx.HTTPError:
        return False, []
    if resp.status_code != 200:
        return False, []
    try:
        body = resp.json()
        models = [m["id"] for m in body.get("data", [])]
    except (KeyError, ValueError):
        return False, []
    return True, models


async def _open_chat_stream(payload: dict) -> httpx.Response:
    req = _client.build_request("POST", "/v1/chat/completions", json=payload)
    try:
        resp = await _client.send(req, stream=True)
    except httpx.HTTPError as exc:
        raise VLLMError(f"request failed: {exc}") from exc

    if resp.status_code >= 500:
        text = (await resp.aread()).decode("utf-8", "replace")[:300]
        await resp.aclose()
        raise VLLMError(f"HTTP {resp.status_code}: {text}")
    if resp.status_code >= 400:
        text = (await resp.aread()).decode("utf-8", "replace")[:300]
        await resp.aclose()
        raise VLLMRejected(f"HTTP {resp.status_code}: {text}")

    return resp


async def stream_chat(payload: dict) -> AsyncIterator[bytes]:
    resp = await _open_chat_stream(payload)
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
    finally:
        await resp.aclose()


async def chat_json(payload: dict) -> dict:
    try:
        resp = await _client.post("/v1/chat/completions", json=payload)
    except httpx.HTTPError as exc:
        raise VLLMError(f"request failed: {exc}") from exc

    if resp.status_code >= 500:
        raise VLLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    if resp.status_code >= 400:
        raise VLLMRejected(f"HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError as exc:
        raise VLLMError(f"malformed json: {exc}") from exc