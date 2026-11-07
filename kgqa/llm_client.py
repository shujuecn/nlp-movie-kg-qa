"""silra API 客户端（OpenAI 兼容）的统一封装。

- api_key 只从环境变量读取（默认 SILRA_API_KEY，可用配置改 key_env）；
  任何文件不得出现明文 key。
- 入口统一调用 load_env_file()：用 python-dotenv 加载根目录 .env（未安装则跳过）。
- chat_json() 走 response_format={"type": "json_object"}，带超时与有限次重试，
  返回解析后的 dict，并累计 token 用量（证据面板展示用）。
- 无 key 的降级路径：调用方先用 has_api_key() 判断，不要在降级分支里构造客户端。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from kgqa.config import LLMConfig


class LLMError(RuntimeError):
    """silra API 调用失败（缺 key、网络错误、输出不合法等）。"""


def load_env_file(dotenv_path: str = ".env") -> bool:
    """加载根目录 .env（python-dotenv）；未安装 dotenv 时静默跳过。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    load_dotenv(dotenv_path)
    return True


def has_api_key(cfg: LLMConfig) -> bool:
    """环境里是否配置了可用的 API key。"""
    return bool(os.environ.get(cfg.key_env, "").strip())


def require_api_key(cfg: LLMConfig) -> str:
    """读取 API key；缺失时给出中文报错提示（extract_llm 等强依赖 API 的入口用）。"""
    key = os.environ.get(cfg.key_env, "").strip()
    if not key:
        raise LLMError(
            f"环境变量 {cfg.key_env} 未配置，无法调用 silra API。"
            "请在仓库根目录的 .env 里填入 key（参考 .env.example），"
            "或先 export SILRA_API_KEY=你的key 再运行。"
        )
    return key


class LLMClient:
    """OpenAI 兼容客户端：JSON 模式对话 + token 用量统计。"""

    def __init__(self, cfg: LLMConfig, api_key: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "未安装 openai 库，无法调用 silra API：pip install openai"
            ) from exc
        self.cfg = cfg
        self._client = OpenAI(
            api_key=api_key, base_url=cfg.base_url, timeout=cfg.timeout, max_retries=0
        )
        self.usage: dict[str, int] = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    @property
    def last_usage(self) -> dict[str, int]:
        """最近累计 token 用量（证据面板展示用）。"""
        return {
            "prompt_tokens": self.usage["prompt_tokens"],
            "completion_tokens": self.usage["completion_tokens"],
        }

    def _chat(
        self, system: str, user: str, *, json_mode: bool, max_tokens: int | None
    ) -> str:
        """对话一次返回原始文本；解析失败或网络错误按配置重试。"""
        limit = max_tokens if max_tokens is not None else self.cfg.max_tokens
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": limit,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                self._track_usage(response)
                return response.choices[0].message.content or ""
            except Exception as exc:  # 网络超时、限流等走重试
                last_error = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(2.0 * (attempt + 1))  # 2s、4s 线性退避
        raise LLMError(
            f"silra API 调用失败（已重试 {self.cfg.max_retries} 次）：{last_error}"
        ) from last_error

    def chat_json(
        self, system: str, user: str, *, max_tokens: int | None = None
    ) -> dict[str, Any]:
        """对话一次并解析 JSON 输出（response_format=json_object）。"""
        content = self._chat(system, user, json_mode=True, max_tokens=max_tokens)
        return _parse_json_object(content)

    def chat_text(
        self, system: str, user: str, *, max_tokens: int | None = None
    ) -> str:
        """对话一次返回纯文本（RAG 生成用，不走 JSON 模式）。"""
        return self._chat(system, user, json_mode=False, max_tokens=max_tokens)

    def _track_usage(self, response: Any) -> None:
        self.usage["requests"] += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage["completion_tokens"] += int(
                getattr(usage, "completion_tokens", 0) or 0
            )


def _parse_json_object(content: str) -> dict[str, Any]:
    """模型输出应为单个 JSON 对象；容忍 ```json 围栏。"""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型输出不是合法 JSON：{content[:200]}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"模型输出应为 JSON 对象，实际是 {type(data).__name__}")
    return data


def create_client(cfg: LLMConfig) -> LLMClient:
    """有 key 时构造客户端；无 key 直接中文报错（不做降级判断用）。"""
    return LLMClient(cfg, require_api_key(cfg))
