"""配置加载。

配置文件是一个 YAML 文件，字段与下面的 dataclass 一一对应。
出现未知字段会直接报错，避免拼写错误被静默忽略。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# 知识图谱实体类型（data/kg/triples.jsonl 的 head_type / tail_type 取值范围）；
# “字面值”用于属性型关系（上映年份/片长/评分/又名）的 tail
ENTITY_TYPES: tuple[str, ...] = (
    "电影",
    "人物",
    "类型",
    "地区",
    "语言",
    "角色",
    "主题",
    "字面值",
)

# build-kg 产出的确定性关系（extract_llm.py 另有 饰演 / 题材）
DETERMINISTIC_RELATIONS: tuple[str, ...] = (
    "导演",
    "编剧",
    "主演",
    "类型",
    "制片地区",
    "语言",
    "上映年份",
    "片长",
    "评分",
    "又名",
)
RELATIONS: tuple[str, ...] = (*DETERMINISTIC_RELATIONS, "饰演", "题材")

# 三元组数据来源：douban-info=结构化字段解析 / llm-offline=离线抽取 / llm-live=课堂现场
TRIPLE_SOURCES: tuple[str, ...] = ("douban-info", "llm-offline", "llm-live")

# 问答路由的问题类型（问题理解 JSON 的 question_type 取值）
QUESTION_TYPES: tuple[str, ...] = ("fact", "overview", "recommend", "out_of_scope")


@dataclass
class DoubanConfig:
    """豆瓣采集与入库（collect_list.py / collect_detail.py）。"""

    search_text: str = "周星驰"  # 搜索列表页关键词
    max_year: int = 2026  # 年份大于该值的未上映条目丢弃
    page_sleep_sec: list[float] = field(default_factory=lambda: [2.0, 3.0])
    detail_sleep_sec: list[float] = field(default_factory=lambda: [3.0, 5.0])
    raw_dir: str = "data/raw"  # 列表清单与单片原始记录
    movies_file: str = "data/movies.jsonl"  # ingest 输出（随仓库分发，进 git）
    # 导演/编剧/主演/配音均无“周星驰”的条目按搜索误命中过滤（可在报告里复核后关闭）
    require_chow_credit: bool = True
    # info 含“首播/集数/季数”行的视为电视剧集，过滤
    drop_tv_series: bool = True


@dataclass
class EmbeddingConfig:
    """SiliconFlow Embedding API（OpenAI 兼容）。
    key 只从环境变量读取，任何文件不得出现明文 key。

    v2.1 起 embedding 全面走 API（Qwen3-Embedding-8B），
    本地不再需要 torch/sentence-transformers。
    """

    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Qwen/Qwen3-Embedding-8B"
    key_env: str = "SILICONFLOW_API_KEY"
    dimensions: int = 1024  # MRL 可调维度（32-4096）；API 返回已归一化向量
    batch_size: int = 32  # 每次请求的文本条数
    timeout: float = 60.0
    max_retries: int = 2


@dataclass
class KgConfig:
    """知识图谱文件（data/kg/ 与评测集随仓库分发）。"""

    dir: str = "data/kg"  # 三元组与别名表目录（随仓库分发，进 git）
    triples_file: str = "data/kg/triples.jsonl"  # 确定性三元组（build-kg 产出）
    triples_llm_file: str = "data/kg/triples_llm.jsonl"  # 离线语义三元组
    aliases_file: str = "data/kg/aliases.jsonl"  # 实体别名表
    eval_file: str = "data/eval/qa_eval.jsonl"  # 30 道带金标的测试题


@dataclass
class RetrieverConfig:
    """FAISS 双索引与实体链接。"""

    synopsis_chunk_chars: int = 300  # 简介分块字数
    synopsis_chunk_overlap: int = 30  # 相邻块重叠字数
    synopsis_top_k: int = 3  # 概述/生成题取多少条简介片段
    entity_top_k: int = 5  # 实体链接向量兜底取多少个候选
    entity_threshold: float = 0.75  # 实体链接向量兜底的余弦阈值
    index_dir: str = "data/index"  # FAISS 索引与元数据目录（可再生产物，gitignore）


@dataclass
class LLMConfig:
    """silra API（OpenAI 兼容）。key 只从环境变量读取，任何文件不得出现明文 key。"""

    base_url: str = "https://api.silra.cn/v1"
    model: str = "qwen3.8-flash"
    key_env: str = "SILRA_API_KEY"
    max_tokens: int = 800  # 单次生成输出 token 上限
    temperature: float = 0.2
    timeout: float = 60.0
    max_retries: int = 2


@dataclass
class GradioConfig:
    """网页工作台。"""

    host: str = "127.0.0.1"
    port: int = 7860


@dataclass
class OutputConfig:
    """评测报告目录（相对仓库根目录；FAISS 索引目录另见 RetrieverConfig.index_dir）。"""

    dir: str = "data"


@dataclass
class Config:
    douban: DoubanConfig = field(default_factory=DoubanConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    kg: KgConfig = field(default_factory=KgConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    gradio: GradioConfig = field(default_factory=GradioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    root: Path = field(default_factory=Path.cwd)

    def validate(self) -> None:
        if not 32 <= self.embedding.dimensions <= 4096:
            raise ValueError(
                f"embedding.dimensions 应在 [32, 4096] 区间，"
                f"实际是 {self.embedding.dimensions}"
            )
        if self.embedding.batch_size < 1:
            raise ValueError("embedding.batch_size 至少为 1")
        if self.embedding.timeout <= 0:
            raise ValueError("embedding.timeout 必须为正数")
        overlap = self.retriever.synopsis_chunk_overlap
        chunk_chars = self.retriever.synopsis_chunk_chars
        if overlap < 0 or overlap >= chunk_chars:
            raise ValueError(
                f"retriever.synopsis_chunk_overlap 需满足 0 <= overlap < chunk_chars"
                f"（当前 overlap={overlap}, chunk_chars={chunk_chars}）"
            )
        if self.retriever.synopsis_top_k < 1:
            raise ValueError("retriever.synopsis_top_k 至少为 1")
        if self.retriever.entity_top_k < 1:
            raise ValueError("retriever.entity_top_k 至少为 1")
        if not 0 < self.retriever.entity_threshold <= 1:
            raise ValueError("retriever.entity_threshold 应在 (0, 1] 区间")
        if self.llm.temperature < 0 or self.llm.temperature > 2:
            raise ValueError("llm.temperature 应在 [0, 2] 区间")
        if self.llm.timeout <= 0:
            raise ValueError("llm.timeout 必须为正数")
        if self.llm.max_retries < 0:
            raise ValueError("llm.max_retries 不能为负数")
        if not 0 < self.gradio.port < 65536:
            raise ValueError(f"gradio.port 端口非法：{self.gradio.port}")
        for group in (self.douban.page_sleep_sec, self.douban.detail_sleep_sec):
            if len(group) != 2 or group[0] < 0 or group[1] < group[0]:
                raise ValueError(
                    "douban.page_sleep_sec / detail_sleep_sec 应是 [下限, 上限] 且非负"
                )

    # ---- 路径均以仓库根目录为基准 ----
    @property
    def artifact_dir(self) -> Path:
        return self.root / self.output.dir

    @property
    def raw_dir(self) -> Path:
        return self.root / self.douban.raw_dir

    @property
    def movies_file(self) -> Path:
        return self.root / self.douban.movies_file

    @property
    def subjects_file(self) -> Path:
        return self.raw_dir / "subjects.json"

    @property
    def kg_dir(self) -> Path:
        return self.root / self.kg.dir

    @property
    def index_dir(self) -> Path:
        return self.root / self.retriever.index_dir

    @property
    def triples_file(self) -> Path:
        return self.root / self.kg.triples_file

    @property
    def triples_llm_file(self) -> Path:
        return self.root / self.kg.triples_llm_file

    @property
    def aliases_file(self) -> Path:
        return self.root / self.kg.aliases_file

    @property
    def eval_file(self) -> Path:
        return self.root / self.kg.eval_file

    @property
    def ingest_report_file(self) -> Path:
        return self.movies_file.parent / "ingest_report.json"

    @property
    def synopsis_index_file(self) -> Path:
        return self.index_dir / "faiss_synopsis.faiss"

    @property
    def synopsis_meta_file(self) -> Path:
        return self.index_dir / "synopsis_meta.jsonl"

    @property
    def entity_index_file(self) -> Path:
        return self.index_dir / "faiss_entity.faiss"

    @property
    def entity_meta_file(self) -> Path:
        return self.index_dir / "entity_meta.jsonl"

    @property
    def eval_report_file(self) -> Path:
        return self.artifact_dir / "eval_report.json"


def _build(cls: type, raw: Any, section: str) -> Any:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置段 {section} 必须是键值表，实际是 {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"配置段 {section} 存在未知字段 {unknown}；可用字段：{sorted(known)}"
        )
    return cls(**raw)


def load_config(
    path: str | Path = "configs/kg_qa.yaml", root: Path | None = None
) -> Config:
    """读取 YAML 配置；root 为仓库根目录，默认当前工作目录。"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件 {config_path}。"
            "请在仓库根目录运行，或用 -c 指定配置文件，例如 -c configs/kg_qa.yaml"
        )

    raw = yaml_safe_load(config_path)
    cfg = Config(
        douban=_build(DoubanConfig, raw.get("douban"), "douban"),
        embedding=_build(EmbeddingConfig, raw.get("embedding"), "embedding"),
        kg=_build(KgConfig, raw.get("kg"), "kg"),
        retriever=_build(RetrieverConfig, raw.get("retriever"), "retriever"),
        llm=_build(LLMConfig, raw.get("llm"), "llm"),
        gradio=_build(GradioConfig, raw.get("gradio"), "gradio"),
        output=_build(OutputConfig, raw.get("output"), "output"),
        root=(root or Path.cwd()).resolve(),
    )
    cfg.validate()
    return cfg


def yaml_safe_load(config_path: Path) -> dict[str, Any]:
    """顶层键值表读取，空文件视为空配置。"""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件 {config_path} 顶层必须是键值表")
    return raw
