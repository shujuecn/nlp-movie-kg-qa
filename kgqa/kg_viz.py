"""图谱可视化辅助（仅供 webapp.py 使用）：邻域提取与 d3 数据序列化。

只依赖 kgqa.graph.load_graph 产出的 networkx.MultiDiGraph（边属性固定
relation / source / head_type / tail_type，见接口契约 §6），不 import、
不改动任何后端模块。所有输出 JSON 面向 gr.HTML 注入，注意体量预算
（单组件注入 ≤200KB，见 PAYLOAD_LIMIT）。
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx

# 实体类型 → 配色（mockup 三轮定稿的 8 色板；胶囊与网络图节点共用）
TYPE_COLORS: dict[str, str] = {
    "电影": "#3b82f6",
    "人物": "#22c55e",
    "角色": "#a855f7",
    "类型": "#f59e0b",
    "地区": "#06b6d4",
    "语言": "#64748b",
    "字面值": "#9ca3af",
    "主题": "#ec4899",
}

FALLBACK_FOCUS = "大话西游之大圣娶亲"
DRAWER_MAX_NODES = 40  # 网络图默认只渲染聚焦实体的 1-2 跳邻域（硬性 ≤40 节点）
MODAL_MAX_NODES = 24  # 证据弹窗逐实体预计算的邻域上限
PAYLOAD_LIMIT = 190_000  # gr.HTML 注入 JSON 的软上限（硬性要求 ≤200KB）


def entity_types(graph: nx.MultiDiGraph) -> dict[str, str]:
    """节点名 → 实体类型。同名多类型时保留先出现的非字面值类型。"""
    types: dict[str, str] = {}
    for u, v, data in graph.edges(data=True):
        for name, ntype in (
            (u, str(data.get("head_type") or "")),
            (v, str(data.get("tail_type") or "")),
        ):
            if not name:
                continue
            current = types.get(name)
            if current is None or (current == "字面值" and ntype and ntype != "字面值"):
                types[name] = ntype or "字面值"
    return types


def graph_stats(
    graph: nx.MultiDiGraph, types: dict[str, str] | None = None
) -> dict[str, int]:
    """三元组条数 / 关系种数 / 实体数 / 电影部数（头部状态签与统计行用）。"""
    if types is None:
        types = entity_types(graph)
    relations = {
        str(data.get("relation") or "") for _, _, data in graph.edges(data=True)
    }
    relations.discard("")
    return {
        "triples": graph.number_of_edges(),
        "relations": len(relations),
        "entities": len(types),
        "movies": sum(1 for ntype in types.values() if ntype == "电影"),
    }


def pick_focus(graph: nx.MultiDiGraph, types: dict[str, str]) -> str:
    """默认聚焦实体：优先《大话西游之大圣娶亲》，否则取度数最高的电影节点。"""
    if FALLBACK_FOCUS in graph:
        return FALLBACK_FOCUS
    best, best_deg = "", -1
    for name, ntype in types.items():
        if ntype == "电影":
            deg = graph.degree(name)
            if deg > best_deg:
                best, best_deg = name, deg
    return best or next(iter(graph.nodes), "")


def neighborhood(
    graph: nx.MultiDiGraph,
    focus: str,
    types: dict[str, str],
    max_nodes: int = DRAWER_MAX_NODES,
    hops: int = 2,
) -> tuple[str, list[dict[str, Any]], list[dict[str, str]]]:
    """从 MultiDiGraph 提取 focus 的 1-2 跳邻域（≤max_nodes 节点）。

    排序规则：实体优先于字面值；同类按度数降序、名字升序（确定性输出）。
    第 2 跳跳过字面值叶子（评分 / 片长之类的节点容易挤占名额）。
    返回 (实际聚焦名, nodes, links)；nodes 保序且 focus 在首位。
    """
    if focus not in graph:
        focus = pick_focus(graph, types)
    if not focus:
        return "", [], []
    degree = {name: int(deg) for name, deg in graph.degree()}
    chosen: dict[str, int] = {focus: 0}  # dict 保序：focus 第一位
    frontier = [focus]
    for depth in range(hops):
        if len(chosen) >= max_nodes:
            break
        candidates: set[str] = set()
        for node in frontier:
            candidates.update(graph.successors(node))
            candidates.update(graph.predecessors(node))
        ranked = sorted(
            (name for name in candidates if name and name not in chosen),
            key=lambda n: (
                1 if types.get(n, "字面值") == "字面值" else 0,
                -degree.get(n, 0),
                n,
            ),
        )
        if depth > 0:
            ranked = [n for n in ranked if types.get(n, "字面值") != "字面值"]
        frontier = []
        for name in ranked:
            if len(chosen) >= max_nodes:
                break
            chosen[name] = len(chosen)
            frontier.append(name)

    links: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for u, v, data in graph.edges(data=True):
        if u in chosen and v in chosen:
            relation = str(data.get("relation") or "")
            key = (u, v, relation)
            if key in seen:
                continue
            seen.add(key)
            links.append(
                {
                    "source": u,
                    "target": v,
                    "relation": relation,
                    "src": str(data.get("source") or ""),
                }
            )
    if len(links) > 300:  # 极端枢纽实体的兜底：优先保留与焦点相连的边
        touching = [
            lk for lk in links if lk["source"] == focus or lk["target"] == focus
        ]
        others = [lk for lk in links if lk not in touching]
        links = (touching + others)[:300]

    nodes = [
        {"id": name, "type": types.get(name, "字面值"), "deg": degree.get(name, 0)}
        for name in chosen
    ]
    return focus, nodes, links


def neighborhood_payload(
    graph: nx.MultiDiGraph,
    focus: str,
    max_nodes: int = DRAWER_MAX_NODES,
    types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """组装单个聚焦实体的 d3 网络图数据（节点 / 边 / 图例 / 配色）。"""
    if types is None:
        types = entity_types(graph)
    focus, nodes, links = neighborhood(graph, focus, types, max_nodes=max_nodes)
    legend: dict[str, int] = {}
    for node in nodes:
        legend[node["type"]] = legend.get(node["type"], 0) + 1
    return {
        "focus": focus,
        "nodes": nodes,
        "links": links,
        "colors": TYPE_COLORS,
        "legend": [
            {"type": ntype, "count": count}
            for ntype, count in sorted(legend.items(), key=lambda kv: -kv[1])
        ],
        "n_nodes": len(nodes),
        "n_links": len(links),
    }


def payloads_for_entities(
    graph: nx.MultiDiGraph,
    entities: list[str],
    max_nodes: int = MODAL_MAX_NODES,
    limit: int = 10,
) -> dict[str, dict[str, Any]]:
    """证据面板弹窗用的逐实体邻域（非字面值实体，按出现顺序限量）。"""
    types = entity_types(graph)
    out: dict[str, dict[str, Any]] = {}
    for name in entities:
        if len(out) >= limit:
            break
        if not name or name in out or name not in graph:
            continue
        if types.get(name, "字面值") == "字面值":
            continue
        out[name] = neighborhood_payload(graph, name, max_nodes=max_nodes, types=types)
    return out


def table_rows(graph: nx.MultiDiGraph) -> list[list[str]]:
    """三元组数据表行（头实体 / 关系 / 尾实体 / 来源），按文件载入顺序。"""
    return [
        [str(u), str(data.get("relation") or ""), str(v), str(data.get("source") or "")]
        for u, v, data in graph.edges(data=True)
    ]


def drawer_payload(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """图谱浏览抽屉的全量数据：边表 + 实体类型 + 统计（前端自行取邻域）。

    体积超 PAYLOAD_LIMIT 时从尾部截边并打 truncated 标记（正常规模不会触发）。
    """
    types = entity_types(graph)
    edges = [[u, r, v, s] for u, r, v, s in table_rows(graph)]
    payload: dict[str, Any] = {
        "focus": pick_focus(graph, types),
        "edges": edges,
        "types": types,
        "colors": TYPE_COLORS,
        "stats": graph_stats(graph, types=types),
        "max_nodes": DRAWER_MAX_NODES,
        "hops": 2,
    }
    text = viz_json(payload)
    while len(text) > PAYLOAD_LIMIT and edges:
        edges = edges[: max(0, len(edges) - 64)]
        payload["edges"] = edges
        payload["truncated"] = True
        payload["stats"]["triples"] = len(edges)
        text = viz_json(payload)
    if payload.get("truncated"):
        print(f"警告：图谱数据超过注入上限，已截断至 {len(edges)} 条边。")
    return payload


def error_payload(message: str) -> dict[str, Any]:
    """图谱未加载时抽屉的占位数据（前端在图区与表格区展示错误说明）。"""
    return {
        "focus": "",
        "edges": [],
        "types": {},
        "colors": TYPE_COLORS,
        "stats": {"triples": 0, "relations": 0, "entities": 0, "movies": 0},
        "max_nodes": DRAWER_MAX_NODES,
        "hops": 2,
        "error": message,
    }


def viz_json(payload: dict[str, Any]) -> str:
    """紧凑序列化并把 "</" 转义为 JSON 合法的 "<\\/"，避免提前闭合 data 块。

    输出注入 <script type="application/json">（raw text 元素，不解析实体），
    前端 textContent → JSON.parse 可无损还原。
    """
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("</", "<\\/")
