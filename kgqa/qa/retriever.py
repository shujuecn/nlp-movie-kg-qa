"""FAISS 双索引：简介分块检索 + 实体名/别名向量检索（v2.1：向量走 SiliconFlow API）。

- 向量一律来自 embed_client.EmbedClient(cfg.embedding).encode()（契约 §3.5）：
  返回已归一化 float32 [n, dimensions]，IndexFlatIP 的检索分数即余弦相似度；
  本模块不 import sentence_transformers / torch，本地零深度学习依赖。
- 两套索引（契约 §5，落在 data/index/）：
    faiss_synopsis.faiss + synopsis_meta.jsonl：简介按字数切块
      （synopsis_chunk_chars=300、重叠 30），行为 {subject_id, title, chunk}；
      summary 为空的影片跳过并在日志中记录；
    faiss_entity.faiss + entity_meta.jsonl：别名表实体名/别名 + 电影片名 +
      llm 三元组的角色/主题端点，行为 {name, entity, entity_type}。
  两个 meta 文件首行写 {"_index_meta": {"embedding_model": ..., "dimensions": ...}}
  记录向量配置；另有 index_fingerprint.json（与索引同目录，cfg.index_dir）
  记录 movies 数量与 mtime、
  embedding 模型/维度与分块参数。
- __init__ 加载现成索引；发现索引缺失、embedding 模型/维度变更或 movies 指纹
  变化时自动重建（读 data/movies.jsonl）；无法重建时中文报错。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from kgqa.config import Config, EmbeddingConfig
from kgqa.embed_client import EmbedClient
from kgqa.kg_linker import load_aliases

__all__ = ["Retriever"]

# 指纹文件与 meta 首行的键名
FINGERPRINT_FILE = "index_fingerprint.json"
_META_HEADER_KEY = "_index_meta"


class Retriever:
    """简介与实体双索引的构建与检索；__init__ 会加载现成索引或在需要时自动重建。"""

    def __init__(self, cfg: Config, embedding_cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self.embedding_cfg = embedding_cfg
        self._fingerprint_file = cfg.index_dir / FINGERPRINT_FILE
        # 契约 §3.5：缺 key 时 EmbedClient 直接 raise EmbedError
        # （index / ask 入口已做 has_embed_key 预检）
        self._embedder = EmbedClient(embedding_cfg)
        self._synopsis_index: faiss.Index | None = None
        self._synopsis_meta: list[dict[str, Any]] = []
        self._entity_index: faiss.Index | None = None
        self._entity_meta: list[dict[str, str]] = []
        self._load_or_rebuild()

    # ---- 对外接口（契约 §5） ----

    def build(self, movies: list[dict[str, Any]]) -> None:
        """构建两套索引并落盘；指纹一致时跳过重建。"""
        stored = _read_json(self._fingerprint_file)
        if (
            stored is not None
            and self._fingerprint_matches(stored, movies)
            and self._all_files_exist()
        ):
            print(
                "索引指纹与 movies 数据一致，跳过重建"
                "（删除 data/index/ 下索引文件可强制重建）。"
            )
            return

        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)

        chunk_texts, chunk_metas = self._synopsis_rows(movies)
        if chunk_texts:
            vectors = self._encode_all(chunk_texts, "简介分块")
            self._synopsis_index = self._new_index(vectors)
            self._synopsis_meta = chunk_metas
            faiss.write_index(self._synopsis_index, str(self.cfg.synopsis_index_file))
            _write_meta(self.cfg.synopsis_meta_file, chunk_metas, self.embedding_cfg)
            print(f"简介索引：{len(chunk_texts)} 块 → {self.cfg.synopsis_index_file}")
        else:
            self._synopsis_index, self._synopsis_meta = None, []
            print("警告：movies 的 summary 全为空，未构建简介索引。")

        entity_names, entity_metas = self._entity_rows(movies)
        if entity_names:
            vectors = self._encode_all(entity_names, "实体名/别名")
            self._entity_index = self._new_index(vectors)
            self._entity_meta = entity_metas
            faiss.write_index(self._entity_index, str(self.cfg.entity_index_file))
            _write_meta(self.cfg.entity_meta_file, entity_metas, self.embedding_cfg)
            print(
                f"实体索引：{len(entity_names)} 个名称 → {self.cfg.entity_index_file}"
            )
        else:
            self._entity_index, self._entity_meta = None, []
            print("警告：别名表与 movies 均无实体名，未构建实体索引。")

        _write_json(self._fingerprint_file, self._fingerprint(movies))
        print(f"Embedding API 用量：{self._embedder.usage}")

    def search_synopsis(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """简介片段检索，返回 [{subject_id, title, chunk, score}]，按相似度降序。"""
        if self._synopsis_index is None or not self._synopsis_meta:
            return []
        scores, ids = self._search(self._synopsis_index, query, k)
        results: list[dict[str, Any]] = []
        for score, idx in zip(scores, ids, strict=True):
            if 0 <= idx < len(self._synopsis_meta):
                row = dict(self._synopsis_meta[idx])
                row["score"] = float(score)
                results.append(row)
        return results

    def search_entities(self, mention: str, k: int = 5) -> list[tuple[str, float]]:
        """实体名/别名向量检索，返回 [(name, score)]（kg_linker 的向量兜底用）。"""
        if self._entity_index is None or not self._entity_meta:
            return []
        scores, ids = self._search(self._entity_index, mention, k)
        return [
            (self._entity_meta[idx]["name"], float(score))
            for score, idx in zip(scores, ids, strict=True)
            if 0 <= idx < len(self._entity_meta)
        ]

    def encode(self, texts: list[str]) -> np.ndarray:
        """文本 → 归一化 float32 向量矩阵（委托 embed_client，契约 §5 v2.1）。"""
        return self._embedder.encode(texts)

    # ---- 索引构建 ----

    def _new_index(self, vectors: np.ndarray) -> faiss.Index:
        """归一化向量 → IndexFlatIP（内积即余弦相似度）。"""
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)
        return index

    def _encode_all(self, texts: list[str], title: str) -> np.ndarray:
        print(
            f"编码{title}：{len(texts)} 条（model={self.embedding_cfg.model}，"
            f"dimensions={self.embedding_cfg.dimensions}）"
        )
        return self.encode(texts)

    def _synopsis_rows(
        self, movies: list[dict[str, Any]]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """简介 → 分块文本与块元数据；summary 为空的影片跳过并记录。"""
        size = self.cfg.retriever.synopsis_chunk_chars
        overlap = self.cfg.retriever.synopsis_chunk_overlap
        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        skipped: list[str] = []
        for movie in movies:
            summary = str(movie.get("summary") or "").strip()
            title = str(movie.get("title") or "")
            subject_id = str(movie.get("subject_id") or "")
            if not summary:
                skipped.append(title or subject_id or "(无标题)")
                continue
            for chunk in _chunk_text(summary, size, overlap):
                texts.append(chunk)
                metas.append({"subject_id": subject_id, "title": title, "chunk": chunk})
        if skipped:
            preview = "、".join(skipped[:8]) + ("……" if len(skipped) > 8 else "")
            print(f"跳过无简介的影片 {len(skipped)} 部：{preview}")
        return texts, metas

    def _entity_rows(
        self, movies: list[dict[str, Any]]
    ) -> tuple[list[str], list[dict[str, str]]]:
        """实体索引的编码对象：别名表实体名/别名 + 电影片名 + llm 三元组角色/主题端点。

        build_kg 只给「有又名」的电影写别名行，因此电影片名从 movies 兜底补齐；
        角色/主题实体只存在于 kg/triples_llm.jsonl，同样补进索引以便向量兜底。
        """
        rows = load_aliases(self.cfg.aliases_file)  # 缺失时中文报错，提示先 build-kg
        metas: list[dict[str, str]] = []
        seen: set[str] = set()

        def add(name: str, entity: str, entity_type: str) -> None:
            name = name.strip()
            if not name or name in seen:
                return
            seen.add(name)
            metas.append(
                {
                    "name": name,
                    "entity": entity.strip() or name,
                    "entity_type": entity_type,
                }
            )

        for row in rows:
            entity = str(row.get("entity") or "")
            entity_type = str(row.get("entity_type") or "")
            add(entity, entity, entity_type)
            for alias in row.get("aliases") or []:
                add(str(alias), entity, entity_type)

        for movie in movies:
            title = str(movie.get("title") or "")
            add(title, title, "电影")

        llm_file = self.cfg.triples_llm_file
        if llm_file.exists():
            for row in _read_jsonl(llm_file):
                for node, node_type in (
                    (row.get("head"), row.get("head_type")),
                    (row.get("tail"), row.get("tail_type")),
                ):
                    if node_type and node_type != "字面值":
                        add(str(node or ""), str(node or ""), str(node_type))

        return [meta["name"] for meta in metas], metas

    # ---- 检索 ----

    def _search(
        self, index: faiss.Index, query: str, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        k = max(1, min(int(k), int(index.ntotal)))
        query_vector = self.encode([query])
        scores, ids = index.search(query_vector, k)
        return scores[0], ids[0]

    # ---- 加载与指纹 ----

    def _load_or_rebuild(self) -> None:
        """加载现成索引；缺失、embedding 配置变更或 movies 指纹变化时自动重建。"""
        stored = _read_json(self._fingerprint_file)
        if (
            stored is not None
            and self._fingerprint_matches_without_movies(stored)
            and self._all_files_exist()
        ):
            self._load_indexes()
            if self._dims_ok():
                print(f"已加载现成索引（指纹一致，跳过重建）：{self._summary()}")
                return
            print("索引维度与当前 embedding.dimensions 不一致，自动重建……")
        elif self._has_any_file():
            print("索引缺失或与当前 embedding 配置 / movies 数据不一致，自动重建……")
        else:
            print("没有现成索引，开始构建……")

        movies = _read_movies_or_none(self.cfg)
        if movies is None:
            raise RuntimeError(
                f"FAISS 索引缺失且找不到 {self.cfg.movies_file}，无法自动重建；"
                "请先运行 python main.py ingest（生成 movies.jsonl），"
                "再运行 python main.py index"
            )
        self.build(movies)

    def _load_indexes(self) -> None:
        synopsis_ready = (
            self.cfg.synopsis_index_file.exists()
            and self.cfg.synopsis_meta_file.exists()
        )
        if synopsis_ready:
            self._synopsis_index = faiss.read_index(str(self.cfg.synopsis_index_file))
            self._synopsis_meta = _read_meta_rows(self.cfg.synopsis_meta_file)
        entity_ready = (
            self.cfg.entity_index_file.exists() and self.cfg.entity_meta_file.exists()
        )
        if entity_ready:
            self._entity_index = faiss.read_index(str(self.cfg.entity_index_file))
            self._entity_meta = _read_meta_rows(self.cfg.entity_meta_file)

    def _dims_ok(self) -> bool:
        dims = int(self.embedding_cfg.dimensions)
        return all(
            index is None or int(index.d) == dims
            for index in (self._synopsis_index, self._entity_index)
        )

    def _has_any_file(self) -> bool:
        return any(
            path.exists()
            for path in (
                self.cfg.synopsis_index_file,
                self.cfg.synopsis_meta_file,
                self.cfg.entity_index_file,
                self.cfg.entity_meta_file,
            )
        )

    def _all_files_exist(self) -> bool:
        return all(
            path.exists()
            for path in (
                self.cfg.synopsis_index_file,
                self.cfg.synopsis_meta_file,
                self.cfg.entity_index_file,
                self.cfg.entity_meta_file,
            )
        )

    def _fingerprint(self, movies: list[dict[str, Any]]) -> dict[str, Any]:
        """重建判据：embedding 模型/维度 + 分块参数 + movies 数量与数据文件 mtime。"""
        return {
            "embedding_model": self.embedding_cfg.model,
            "dimensions": int(self.embedding_cfg.dimensions),
            "chunk_chars": self.cfg.retriever.synopsis_chunk_chars,
            "chunk_overlap": self.cfg.retriever.synopsis_chunk_overlap,
            "movies_count": len(movies),
            "movies_mtime": _movies_mtime(self.cfg.movies_file),
        }

    def _fingerprint_matches(
        self, stored: dict[str, Any], movies: list[dict[str, Any]]
    ) -> bool:
        return stored == self._fingerprint(movies)

    def _fingerprint_matches_without_movies(self, stored: dict[str, Any]) -> bool:
        """__init__ 阶段手里没有 movies 列表，只比对模型/维度/分块参数与 mtime。"""
        expected = self._fingerprint([])
        for key in ("embedding_model", "dimensions", "chunk_chars", "chunk_overlap"):
            if stored.get(key) != expected[key]:
                return False
        movies_mtime = _movies_mtime(self.cfg.movies_file)
        # movies 文件缺失时无法核对 mtime，视作未变化（仅按模型/维度/分块参数判断）
        return movies_mtime is None or stored.get("movies_mtime") == movies_mtime

    def _summary(self) -> str:
        chunks = self._synopsis_index.ntotal if self._synopsis_index is not None else 0
        entities = self._entity_index.ntotal if self._entity_index is not None else 0
        return f"简介 {chunks} 块 + 实体名称 {entities} 个"


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """按字数切块（带重叠）；尾块若完全落在前一块的重叠区则丢弃。"""
    if len(text) <= size:
        return [text]
    step = max(size - overlap, 1)
    starts = list(range(0, len(text), step))
    if len(starts) > 1 and len(text) - starts[-1] <= overlap:
        starts.pop()
    return [text[start : start + size] for start in starts]


def _movies_mtime(movies_file: Path) -> int | None:
    return int(movies_file.stat().st_mtime) if movies_file.exists() else None


def _read_movies_or_none(cfg: Config) -> list[dict[str, Any]] | None:
    """自动重建用：movies.jsonl 不存在返回 None；存在则逐行解析（空文件当空列表）。"""
    if not cfg.movies_file.exists():
        return None
    return [
        json.loads(line)
        for line in cfg.movies_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_meta(
    path: Path, rows: list[dict[str, Any]], embedding_cfg: EmbeddingConfig
) -> None:
    """meta jsonl：首行 _index_meta（embedding 模型与维度），其余为数据行。"""
    header = {
        _META_HEADER_KEY: {
            "embedding_model": embedding_cfg.model,
            "dimensions": int(embedding_cfg.dimensions),
        }
    }
    with Path(path).open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_meta_rows(path: Path) -> list[dict[str, Any]]:
    """读取 meta jsonl 的数据行（跳过首行 _index_meta）。"""
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if _META_HEADER_KEY in row:
            continue
        rows.append(row)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    """检索排序演示（纯读）：python -m kgqa.qa.retriever "查询词" [-k N]"""
    import argparse

    from kgqa.config import load_config
    from kgqa.llm_client import load_env_file

    parser = argparse.ArgumentParser(description="FAISS 简介检索排序演示（只读）")
    parser.add_argument(
        "query",
        nargs="?",
        default="介绍一下《功夫》讲了什么故事。",
        help="查询文本",
    )
    parser.add_argument("-k", type=int, default=10, help="展示前多少条")
    parser.add_argument("-c", "--config", default="configs/kg_qa.yaml", help="配置文件")
    args = parser.parse_args()

    load_env_file()
    demo_cfg = load_config(args.config)
    demo_retriever = Retriever(demo_cfg, demo_cfg.embedding)
    hits = demo_retriever.search_synopsis(args.query, k=args.k)
    print(f"查询：{args.query}")
    for rank, hit in enumerate(hits, 1):
        print(f"{rank:2d}. [{hit['title']}] score={hit['score']:.4f}")
        print(f"    {hit['chunk'][:38]}")
