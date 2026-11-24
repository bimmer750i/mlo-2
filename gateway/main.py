"""Real-time inference gateway — СКЕЛЕТ.

Готовы /health и /metrics. Остальные ручки реализуйте сами по контракту
(API_CONTRACT.md в корне этого каталога):
    /triton/health, /vllm/health, /metrics/triton, /metrics/vllm, /metrics/node,
    /info (Bearer, с полем gpu), /vision (Bearer, модель вашего варианта — make variant),
    /chat (Bearer, SSE).

Как структурировать код (auth, конфиг, инференс) — на ваше усмотрение.
"""
from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Real-time Inference Gateway")

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator

import metrics_proxy
import triton_client
import vllm_client
from auth import require_auth
from config import settings
from preprocessing import InvalidImageError, preprocess_image
from schemas import ChatRequest, Prediction, VisionResponse
from utils import detect_gpu, load_labels, softmax_top5


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await triton_client.close()
    await vllm_client.close()
    await metrics_proxy.close()


app = FastAPI(title="Real-time Inference Gateway", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Контракт: /chat при битом JSON -> 400 (FastAPI по умолчанию даёт 422).
    return JSONResponse(status_code=400, content={"detail": "invalid request"})


GPU_NAME = detect_gpu()
LABELS = load_labels()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/triton/health")
async def triton_health() -> JSONResponse:
    ready = await triton_client.is_ready()
    body = {"backend": "triton", "ready": ready, "model": settings.cv_model}
    return JSONResponse(status_code=200 if ready else 503, content=body)


@app.get("/vllm/health")
async def vllm_health() -> JSONResponse:
    ready, models = await vllm_client.is_ready()
    body = {"backend": "vllm", "ready": ready, "models": models}
    return JSONResponse(status_code=200 if ready else 503, content=body)

@app.get("/metrics/triton")
async def metrics_triton():
    return await metrics_proxy.triton_metrics()

@app.get("/metrics/vllm")
async def metrics_vllm():
    return await metrics_proxy.vllm_metrics()


@app.get("/metrics/node")
async def metrics_node():
    return await metrics_proxy.node_metrics()


@app.get("/info")
def info(_: dict = require_auth) -> dict:
    return {
        "student": {"name": settings.student_name, "email": settings.student_email},
        "models": {"vision": settings.cv_model, "llm": settings.llm_model},
        "gpu": GPU_NAME,
        "version": settings.version,
    }

@app.post("/vision", response_model=VisionResponse)
async def vision(
    file: UploadFile = File(...),
    _: dict = require_auth,
) -> VisionResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        x = preprocess_image(raw)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=f"not an image: {exc}")

    t0 = time.perf_counter()
    try:
        logits = await triton_client.infer(x)
    except triton_client.TritonError as exc:
        raise HTTPException(status_code=502, detail=f"triton: {exc}")
    latency_ms = (time.perf_counter() - t0) * 1000.0

    predictions = [
        Prediction(class_id=cid, label=LABELS[cid], score=score)
        for cid, score in softmax_top5(logits)
    ]
    return VisionResponse(
        model=settings.cv_model,
        predictions=predictions,
        latency_ms=round(latency_ms, 2),
    )

@app.post("/chat")
async def chat(req: ChatRequest, _: dict = require_auth):
    payload: dict = {
        "model": settings.llm_model,
        "messages": req.messages,
        "stream": req.stream,
    }
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        payload["temperature"] = req.temperature

    if not req.stream:
        try:
            return await vllm_client.chat_json(payload)
        except vllm_client.VLLMRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except vllm_client.VLLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    try:
        resp = await vllm_client.open_chat_stream(payload)
    except vllm_client.VLLMRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except vllm_client.VLLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    async def gen():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream")
