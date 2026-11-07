"""SiliconFlow Embedding API 客户端（OpenAI 兼容 /v1/embeddings）。

v2.1 起 embedding 全面走 API（Qwen3-Embedding-8B），本地不再需要
torch / sentence-transformers。要点：

- key 只从环境变量读取（默认 SILICONFLOW_API_KEY，可用配置改 key_env）；
  任何文件不得出现明文 key；入口统一先调 llm_client.load_env_file() 加载 .env。
- encode() 自动按 cfg.batch_size 分批请求，返回 float32 归一化矩阵 [n, dimensions]
  （API 本身已归一化，这里再除一次范数做防御，保证内积即余弦）。
- 调用方先用 has_embed_key() 判断，缺 key 时走降级路径，不要构造客户端。
"""

from __future__ import annotations

import os
import time

import numpy as np

from kgqa.config import EmbeddingConfig


class EmbedError(RuntimeError):
    """Embedding API 调用失败（缺 key、网络错误、返回不合法等）。"""


def has_embed_key(cfg: EmbeddingConfig) -> bool:
    """环境里是否配置了可用的 Embedding API key。"""
    return bool(os.environ.get(cfg.key_env, "").strip())


def require_embed_key(cfg: EmbeddingConfig) -> str:
    """读取 key；缺失时给出中文报错提示（index 等强依赖 embedding 的入口用）。"""
    key = os.environ.get(cfg.key_env, "").strip()
    if not key:
        raise EmbedError(
            f"环境变量 {cfg.key_env} 未配置，无法调用 Embedding API。"
            "请在仓库根目录的 .env 里填入 SiliconFlow 的 key（参考 .env.example），"
            "或先 export SILICONFLOW_API_KEY=你的key 再运行。"
        )
    return key


class EmbedClient:
    """OpenAI 兼容 Embedding 客户端：批量编码 + 归一化。"""

    def __init__(self, cfg: EmbeddingConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbedError(
                "未安装 openai 库，无法调用 Embedding API：pip install openai"
            ) from exc
        self.cfg = cfg
        self._client = OpenAI(
            api_key=require_embed_key(cfg),
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            max_retries=0,
        )
        self.usage: dict[str, int] = {"requests": 0, "prompt_tokens": 0}

    def encode(self, texts: list[str]) -> np.ndarray:
        """批量编码为归一化 float32 矩阵 [n, dimensions]；空列表返回 (0, dims)。"""
        if not texts:
            return np.zeros((0, self.cfg.dimensions), dtype="float32")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.cfg.batch_size):
            batch = [str(t) for t in texts[start : start + self.cfg.batch_size]]
            vectors.extend(self._embed_batch(batch))
        arr = np.asarray(vectors, dtype="float32")
        if arr.shape[1] != self.cfg.dimensions:
            raise EmbedError(
                f"API 返回维度 {arr.shape[1]} 与配置 "
                f"dimensions={self.cfg.dimensions} 不一致"
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """单批请求，网络错误按配置重试；按 index 还原顺序。"""
        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                response = self._client.embeddings.create(
                    model=self.cfg.model,
                    input=batch,
                    dimensions=self.cfg.dimensions,
                )
                self.usage["requests"] += 1
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.usage["prompt_tokens"] += int(
                        getattr(usage, "prompt_tokens", 0) or 0
                    )
                data = sorted(response.data, key=lambda item: item.index)
                return [list(map(float, item.embedding)) for item in data]
            except Exception as exc:  # 网络超时、限流走重试
                last_error = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(2.0 * (attempt + 1))  # 2s、4s 线性退避
        raise EmbedError(
            f"Embedding API 调用失败（已重试 {self.cfg.max_retries} 次）：{last_error}"
        ) from last_error


def create_embedder(cfg: EmbeddingConfig) -> EmbedClient:
    """有 key 时构造客户端；无 key 直接中文报错（不做降级判断用）。"""
    return EmbedClient(cfg)
