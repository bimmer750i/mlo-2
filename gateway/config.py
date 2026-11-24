from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    jwt_secret: str = Field(..., alias="JWT_SECRET")

    cv_model: str = Field("efficientnet_b0", alias="CV_MODEL")
    llm_model: str = Field("Qwen2.5-1.5B-Instruct", alias="LLM_MODEL")
    student_name: str = Field("", alias="STUDENT_NAME")
    student_email: str = Field("", alias="STUDENT_EMAIL")

    triton_url: str = Field("http://triton:8000", alias="TRITON_URL")
    triton_metrics_url: str = Field("http://triton:8002", alias="TRITON_METRICS_URL")
    vllm_url: str = Field("http://vllm:8000", alias="VLLM_URL")
    node_exporter_url: str = Field("http://node-exporter:9100", alias="NODE_EXPORTER_URL")

    gpu_name: str = Field("", alias="GPU_NAME")
    version: str = "1.0"

    http_timeout: float = 10.0

    vllm_stream_timeout: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()