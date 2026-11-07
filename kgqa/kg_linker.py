"""实体链接：mention → 图谱实体的三级匹配。

按成本从低到高依次尝试（契约 §4）：

  1. 精确：mention 与实体名或任一别名一致（大小写不敏感，score=1.0，method=exact）；
  2. 子串：双向包含（mention ⊆ 候选 或 候选 ⊆ mention），多个命中取「匹配串最长」者
     （并列时得分高者优先，method=substring），得分 = len(mention) / len(entity)——
     entity 是别名行的规范实体名，mention 比实体名长时得分会大于 1，仅表示覆盖关系；
  3. 向量兜底：调用方传入 vector_fallback（SiliconFlow Embedding + FAISS 余弦
     检索，见 retriever.Retriever.search_entities），top1 ≥ threshold 才采信
     （method=vector）。

别名表来自 data/kg/aliases.jsonl（{entity, entity_type, aliases, source?}），
匹配时实体名与别名同权；向量兜底返回的 name 会回查别名表得到规范实体与类型。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["LinkResult", "link_entity", "load_aliases"]


@dataclass
class LinkResult:
    """一条实体链接结果；method ∈ exact|substring|vector。"""

    entity: str
    entity_type: str
    score: float
    method: str


def load_aliases(path: str | Path) -> list[dict[str, Any]]:
    """读取别名表 jsonl；文件缺失时中文报错（提示先运行 build-kg）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"没有找到别名表 {path}，请先运行：python main.py build-kg"
        )
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def link_entity(
    mention: str,
    aliases: list[dict[str, Any]],
    vector_fallback: Callable[[str], list[tuple[str, float]]] | None = None,
    threshold: float = 0.75,
) -> LinkResult | None:
    """单个 mention 的三级实体链接，链接失败返回 None。

    vector_fallback 形如 retriever.Retriever.search_entities：
    mention → [(name, score)]，即「候选名与相似度」列表。
    """
    mention = (mention or "").strip()
    if not mention:
        return None

    # 一级：实体名 / 别名精确命中（大小写不敏感，覆盖英文名），按登记顺序取第一个
    for row in aliases:
        for candidate in _row_names(row):
            if candidate.casefold() == mention.casefold():
                return LinkResult(
                    _entity_of(row, candidate), _type_of(row), 1.0, "exact"
                )

    # 二级：双向子串，取匹配串最长者（并列取得分高者），整体保持登记顺序稳定
    best_len, best_score, best_result = -1, -1.0, None
    for row in aliases:
        entity = str(row.get("entity") or "").strip()
        for candidate in _row_names(row):
            if candidate == mention:  # 精确阶段已处理
                continue
            if candidate not in mention and mention not in candidate:
                continue
            score = len(mention) / max(len(entity), 1)
            better = len(candidate) > best_len or (
                len(candidate) == best_len and score > best_score
            )
            if better:
                best_len, best_score = len(candidate), score
                best_result = LinkResult(
                    entity or candidate, _type_of(row), score, "substring"
                )
    if best_result is not None:
        return best_result

    # 三级：向量兜底，top1 ≥ threshold 才采信
    if vector_fallback is not None:
        hits = vector_fallback(mention)
        if hits:
            name, score = hits[0]
            if float(score) >= threshold:
                matched = _find_row(aliases, str(name))
                if matched is not None:
                    return LinkResult(
                        _entity_of(matched, str(name)),
                        _type_of(matched),
                        float(score),
                        "vector",
                    )
                return LinkResult(str(name).strip(), "", float(score), "vector")
    return None


def _row_names(row: dict[str, Any]) -> list[str]:
    """一个别名表行的全部候选：实体名 + 别名（同权、去重、去首尾空白）。"""
    names = [str(row.get("entity") or ""), *(str(a) for a in row.get("aliases") or [])]
    return [name.strip() for name in dict.fromkeys(names) if name.strip()]


def _entity_of(row: dict[str, Any], matched: str) -> str:
    return str(row.get("entity") or "").strip() or matched


def _type_of(row: dict[str, Any]) -> str:
    return str(row.get("entity_type") or "")


def _find_row(aliases: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """向量兜底命中的 name 回查别名表（实体名或别名），换回规范实体与类型。"""
    for row in aliases:
        if name in _row_names(row):
            return row
    return None
