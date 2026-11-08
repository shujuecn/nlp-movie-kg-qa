"""三元组文件 → NetworkX 图谱加载。

从 kg_qa.py 拆出（v2.2 模块化）：图谱加载是 build_kg（写）与问答（读）
共用的边界模块，单独成文件以便「图谱浏览」等模块直接复用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx


def load_graph(kg_dir: str | Path) -> nx.MultiDiGraph:
    """读取 kg 目录（data/kg/）下三元组构建 MultiDiGraph；triples_llm.jsonl 存在才合并。

    边属性固定 relation / source / head_type / tail_type（llm 行的 subject_id
    是 extract_llm.py 去重用的附加字段，不进入图）。
    """
    kg_dir = Path(kg_dir)
    triples_file = kg_dir / "triples.jsonl"
    if not triples_file.exists():
        raise FileNotFoundError(
            f"没有找到三元组文件 {triples_file}，请先运行：python main.py build-kg"
        )

    graph = nx.MultiDiGraph()
    total = 0
    for path in (triples_file, kg_dir / "triples_llm.jsonl"):
        if not path.exists():
            continue
        added = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            row = _parse_triple_row(line)
            if row is None:
                continue
            graph.add_edge(
                str(row["head"]),
                str(row["tail"]),
                relation=str(row.get("relation", "")),
                source=str(row.get("source", "")),
                head_type=str(row.get("head_type", "")),
                tail_type=str(row.get("tail_type", "")),
            )
            added += 1
        total += added
        print(f"  载入 {path.name}：{added} 条三元组")
    print(f"知识图谱：{graph.number_of_nodes()} 个节点 / {total} 条边")
    return graph


def _parse_triple_row(line: str) -> dict[str, Any] | None:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        print(f"  警告：跳过无法解析的三元组行：{line[:60]}")
        return None
    if not isinstance(row, dict) or not row.get("head") or not row.get("tail"):
        print(f"  警告：跳过缺少 head/tail 的三元组行：{line[:60]}")
        return None
    return row
