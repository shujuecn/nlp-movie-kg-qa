"""问答编排模块：路由 → 事实模板直答 / RAG 组装生成 / 降级与兜底。

从 kg_qa.py 拆出（v2.2 模块化）。路由与模式（契约 §6）：
  fact         图谱谓词查询 → 模板直答（本地、零成本、毫秒级）     mode=direct
  overview     子图三元组 + 简介片段 → API 生成（要求引用 [1][2]）  mode=rag
  recommend    同上，提示词换成推荐语气                            mode=rag
  out_of_scope 本地礼貌兜底                                        mode=fallback
  无 SILRA key 的生成题：跳过生成，返回检索证据摘要 + 配置提示        mode=degraded
  retriever 为 None（无 SILICONFLOW key）：overview/recommend 直接 degraded，
  passages 为空，evidence.understanding.degraded_reason 说明原因。

evidence 固定四个键：understanding（问题理解 + 链接结果）/ triples / passages /
api_usage。生成类证据编号：三元组 [1..T]、简介片段 [T+1..T+P]，
evaluate.py 据此校验引用有效性。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import networkx as nx

from kgqa.config import RELATIONS, Config
from kgqa.kg_linker import LinkResult, link_entity
from kgqa.llm_client import LLMClient
from kgqa.qa.retriever import Retriever
from kgqa.qa.understanding import relations_from_question, understand_question

__all__ = ["answer"]

# 电影等实体作 head（被指向属性/人物）时的关系句式；上映年份/又名/饰演单独处理
_MOVIE_RELATION_LABELS: dict[str, str] = {
    "导演": "导演",
    "编剧": "编剧",
    "主演": "主演",
    "配音": "配音",
    "类型": "类型",
    "制片地区": "制片地区",
    "语言": "语言",
    "片长": "片长",
    "评分": "评分",
    "题材": "题材",
}
# 人物/类型/地区等作 tail（指向别人）时的反向句式
_SUBJECT_PHRASES: dict[str, str] = {
    "导演": "执导的电影",
    "编剧": "担任编剧的电影",
    "主演": "主演的电影",
    "配音": "配音的电影",
    "饰演": "饰演过的角色",
    "类型": "类型的电影",
    "制片地区": "出品的电影",
    "语言": "对白语言的电影",
    "题材": "题材的电影",
}

OUT_OF_SCOPE_REPLY = (
    "这个问题超出了我的能力范围。我是周星驰电影知识图谱问答系统，可以回答"
    "周星驰电影的导演、主演、评分、上映年份、类型、又名、角色饰演关系，"
    "以及剧情概述和电影推荐。欢迎换个问题试试。"
)
NO_ENTITY_REPLY = (
    "没有在知识图谱中找到与问题相关的实体，无法直接作答。"
    "可以试试：功夫是谁导演的？紫霞仙子是谁演的？《食神》的评分是多少？"
)
NO_EVIDENCE_REPLY = (
    "没有在知识图谱和简介库中检索到与该问题相关的证据，暂时无法回答。"
    "可以试试问某部电影的导演、主演、评分、上映年份，或某个角色由谁饰演。"
)

RAG_SYSTEM = """\
你是周星驰电影知识图谱问答系统的「回答生成」模块。请严格只依据「证据」回答用户问题：
- 每个关键事实后面标注证据编号，如 [1]、[2]，一条证据可以多次引用；
- 不要编造证据之外的信息，证据不足时明确说明依据现有资料无法回答；
- 用中文回答，通顺简洁，控制在 200 字以内。
"""
RAG_RECOMMEND_EXTRA = (
    "- 这是推荐类问题：用短列表给出推荐影片，每部配一句推荐理由（引用证据）。\n"
)


def answer(
    question: str,
    *,
    graph: nx.MultiDiGraph,
    aliases: list[dict[str, Any]],
    retriever: Retriever | None,
    llm: LLMClient | None,
    cfg: Config,
    force_route: str | None = None,
) -> dict[str, Any]:
    """单题问答主流程；返回 {"mode", "answer", "evidence"}（契约 §6）。

    force_route（webapp 回答模式用，默认 None 走智能路由）：
      "fact"     强制模板直答；"rag" 强制生成路线；"evidence" 只返回检索证据不生成
      （mode=degraded，零 API）；"fallback" 保留给超范围。
    retriever 可为 None（未配 SILICONFLOW_API_KEY）：fact / out_of_scope 不受影响，
    overview / recommend 直接 mode=degraded。
    """
    understanding = understand_question(llm, cfg, question)
    linked = _link_mentions(
        understanding["entity_mentions"],
        _effective_aliases(aliases, graph),
        retriever,
        cfg,
    )
    understanding_evidence = {
        **understanding,
        "linked": [
            {
                "entity": r.entity,
                "entity_type": r.entity_type,
                "score": r.score,
                "method": r.method,
            }
            for r in linked
        ],
    }

    question_type = understanding["question_type"]
    route = force_route or {"out_of_scope": "fallback", "fact": "fact"}.get(
        question_type, "rag"
    )
    if route == "fallback":
        return {
            "mode": "fallback",
            "answer": OUT_OF_SCOPE_REPLY,
            "evidence": _evidence(understanding_evidence, [], [], _api_usage(llm)),
        }
    if route == "evidence":
        # 仅检索模式（用户指定）：组装证据但不生成，零 API 成本
        triples = _subgraph_triples(graph, linked)
        passages = _collect_passages(retriever, question, linked, cfg)
        understanding = {
            **understanding_evidence,
            "degraded_reason": "仅检索模式（用户指定）：只返回证据，不生成答案",
        }
        return {
            "mode": "degraded",
            "answer": _degraded_answer(
                triples, passages, ["仅检索模式（用户指定）"], []
            ),
            "evidence": _evidence(understanding, triples, passages, None),
        }
    if route == "fact":
        return _answer_fact(
            question,
            graph,
            linked,
            understanding["target_relations"],
            understanding_evidence,
        )
    return _answer_rag(
        question,
        graph,
        linked,
        question_type,
        understanding_evidence,
        retriever,
        llm,
        cfg,
    )


# ---- 实体链接 ----


def _link_mentions(
    mentions: list[str],
    aliases: list[dict[str, Any]],
    retriever: Retriever | None,
    cfg: Config,
) -> list[LinkResult]:
    """逐个 mention 链接实体；向量兜底走 retriever.search_entities，同一实体去重。"""
    threshold = cfg.retriever.entity_threshold
    top_k = cfg.retriever.entity_top_k

    # 向量兜底仅在 retriever 可用时提供；link_entity 只在兜底非 None 时才调用它
    fallback: Callable[[str], list[tuple[str, float]]] | None = None
    if retriever is not None:

        def fallback(mention: str) -> list[tuple[str, float]]:
            return retriever.search_entities(mention, k=top_k)

    results: list[LinkResult] = []
    seen: set[str] = set()
    for mention in mentions:
        result = link_entity(
            mention, aliases, vector_fallback=fallback, threshold=threshold
        )
        if result is None or result.entity in seen:
            continue
        seen.add(result.entity)
        results.append(result)
    return results


def _effective_aliases(
    aliases: list[dict[str, Any]], graph: nx.MultiDiGraph
) -> list[dict[str, Any]]:
    """别名表 + 图谱端点实体（角色/主题/无别名行的电影等），保证链接三级全覆盖。

    向量兜底返回的 name 也按这份列表回查规范实体与类型，因此实体索引里的
    每个名字（别名表、电影片名、llm 三元组端点）都能在这里找到归属行。
    """
    known = {str(row.get("entity") or "").strip() for row in aliases}
    types: dict[str, str] = {}
    for u, v, data in graph.edges(data=True):
        for node, node_type in ((u, data.get("head_type")), (v, data.get("tail_type"))):
            name = str(node or "").strip()
            ntype = str(node_type or "").strip()
            if not name or not ntype or ntype == "字面值":
                continue
            types.setdefault(name, ntype)
    extra = [
        {"entity": name, "entity_type": ntype, "aliases": []}
        for name, ntype in sorted(types.items())
        if name not in known
    ]
    return [*aliases, *extra]


def _edge_row(u: str, v: str, data: dict[str, Any]) -> dict[str, Any]:
    """networkx 边 → 契约 §2 的三元组行（evaluate 与 webapp 消费的统一形状）。"""
    return {
        "head": u,
        "head_type": str(data.get("head_type") or ""),
        "relation": str(data.get("relation") or ""),
        "tail": v,
        "tail_type": str(data.get("tail_type") or ""),
        "source": str(data.get("source") or ""),
    }


# ---- fact：图谱谓词查询 + 模板直答 ----


def _answer_fact(
    question: str,
    graph: nx.MultiDiGraph,
    linked: list[LinkResult],
    target_relations: list[str],
    understanding: dict[str, Any],
) -> dict[str, Any]:
    if not linked:
        return {
            "mode": "direct",
            "answer": NO_ENTITY_REPLY,
            "evidence": _evidence(understanding, [], [], _api_usage(llm=None)),
        }

    # 「合作过哪些电影」：多实体求交集，属于 fact 直答（契约正文示例）
    if len(linked) >= 2 and ("合作" in question or "一起" in question):
        return _answer_collaboration(graph, linked, understanding)

    known_relations = [r for r in target_relations if r in RELATIONS]
    relations = known_relations or relations_from_question(question)
    if relations:
        edges = _collect_edges(graph, linked, relations)
        if edges:
            answer_text = _format_fact(linked, edges, relations)
            return {
                "mode": "direct",
                "answer": answer_text,
                "evidence": _evidence(understanding, edges, [], _api_usage(llm=None)),
            }

    # 目标关系未命中或没识别出关系 → 输出实体档案（仍是本地直答）
    edges = _collect_edges(graph, linked, [])
    if edges:
        answer_text = _format_profile(linked, edges)
    else:
        names = "、".join(_display_name(r) for r in linked)
        answer_text = f"知识图谱中没有查到「{names}」的相关事实。"
    return {
        "mode": "direct",
        "answer": answer_text,
        "evidence": _evidence(understanding, edges, [], _api_usage(llm=None)),
    }


def _answer_collaboration(
    graph: nx.MultiDiGraph, linked: list[LinkResult], understanding: dict[str, Any]
) -> dict[str, Any]:
    """多实体合作题：求各实体参与电影集合的交集，直答列举。"""
    crew_relations = {"导演", "编剧", "主演", "配音"}
    movie_sets: list[set[str]] = []
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for result in linked:
        movies: set[str] = set()
        for u, v, data in list(graph.out_edges(result.entity, data=True)) + list(
            graph.in_edges(result.entity, data=True)
        ):
            if data.get("relation") not in crew_relations:
                continue
            key = (u, str(data.get("relation")), v)
            if key in seen:
                continue
            seen.add(key)
            if u == result.entity:
                other, other_type = v, data.get("tail_type")
            else:
                other, other_type = u, data.get("head_type")
            if other_type == "电影":
                movies.add(str(other))
                edges.append(_edge_row(u, v, data))
        movie_sets.append(movies)

    common: set[str] = set()
    if movie_sets and all(movie_sets):
        common = set.intersection(*movie_sets)

    names = "与".join(_display_name(r) for r in linked)
    if common:
        listed = "、".join(f"《{m}》" for m in sorted(common))
        answer_text = f"{names}合作过的电影有：{listed}。"
    else:
        answer_text = f"知识图谱中没有查到{names}合作过的电影。"

    relevant = [
        row
        for row in edges
        if (row["head"] if row["head_type"] == "电影" else row["tail"]) in common
    ][:40]
    return {
        "mode": "direct",
        "answer": answer_text,
        "evidence": _evidence(understanding, relevant, [], _api_usage(llm=None)),
    }


def _collect_edges(
    graph: nx.MultiDiGraph, linked: list[LinkResult], relations: list[str]
) -> list[dict[str, Any]]:
    """链接实体的 1-hop 边（relations 为空取全部），按三元组键去重。"""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for result in linked:
        for u, v, data in list(graph.out_edges(result.entity, data=True)) + list(
            graph.in_edges(result.entity, data=True)
        ):
            relation = str(data.get("relation") or "")
            if relations and relation not in relations:
                continue
            key = (u, relation, v)
            if key in seen:
                continue
            seen.add(key)
            rows.append(_edge_row(u, v, data))
    return rows


def _format_fact(
    linked: list[LinkResult], edges: list[dict[str, Any]], relations: list[str]
) -> str:
    """按（实体 × 关系 × 方向）分组套直答模板，target_relations 优先。"""
    sentences: list[str] = []
    for result in linked:
        mine = [e for e in edges if result.entity in (e["head"], e["tail"])]
        order = [r for r in relations if r in {e["relation"] for e in mine}]
        order += [r for r in _relation_order(mine) if r not in order]
        for relation in order:
            out_rows = [
                e
                for e in mine
                if e["head"] == result.entity and e["relation"] == relation
            ]
            in_rows = [
                e
                for e in mine
                if e["tail"] == result.entity and e["relation"] == relation
            ]
            if out_rows:
                sentences.append(_movie_side_sentence(result, relation, out_rows))
            if in_rows:
                sentences.append(_subject_side_sentence(result, relation, in_rows))
    return "".join(sentences) or "知识图谱中没有查到相关事实。"


def _format_profile(linked: list[LinkResult], edges: list[dict[str, Any]]) -> str:
    """目标关系查无时输出实体档案：《X》：导演：周星驰；主演：…；上映于 2004 年；…。"""
    parts: list[str] = []
    for result in linked:
        mine = [e for e in edges if result.entity in (e["head"], e["tail"])][:40]
        clauses: list[str] = []
        for relation in _relation_order(mine):
            out_rows = [
                e
                for e in mine
                if e["head"] == result.entity and e["relation"] == relation
            ]
            in_rows = [
                e
                for e in mine
                if e["tail"] == result.entity and e["relation"] == relation
            ]
            if out_rows:
                clauses.append(_profile_clause(relation, out_rows, "out"))
            if in_rows:
                clauses.append(_profile_clause(relation, in_rows, "in"))
        if clauses:
            parts.append(f"{_display_name(result)}：{'；'.join(clauses)}。")
    return "".join(parts) or "知识图谱中没有查到相关事实。"


def _movie_side_sentence(
    result: LinkResult, relation: str, rows: list[dict[str, Any]]
) -> str:
    """实体作 head（电影属性、电影→人物、角色→演员）时的直答句。"""
    subject = _display_name(result)
    items = [(str(row["tail"]), str(row["tail_type"])) for row in rows]
    if relation == "上映年份":
        return f"{subject}上映于{items[0][0]}年。"
    if relation == "饰演":
        return f"角色{subject}由{_join(items)}饰演。"
    if relation == "又名":
        return f"{subject}又名{_join(items)}。"
    label = _MOVIE_RELATION_LABELS.get(relation, relation)
    if len(items) == 1:
        return f"{subject}的{label}是{_join(items)}。"
    return f"{subject}的{label}有{_join(items)}。"


def _subject_side_sentence(
    result: LinkResult, relation: str, rows: list[dict[str, Any]]
) -> str:
    """实体作 tail（人物/类型/地区/语言等被指向）时的直答句。"""
    subject = _display_name(result)
    items = [(str(row["head"]), str(row["head_type"])) for row in rows]
    phrase = _SUBJECT_PHRASES.get(relation, f"相关的{relation}条目")
    return f"{subject}{phrase}有：{_join(items)}。"


def _profile_clause(relation: str, rows: list[dict[str, Any]], side: str) -> str:
    """档案句里的单个分句，如「导演：周星驰」「上映于 2004 年」。"""
    if side == "out":
        items = [(str(row["tail"]), str(row["tail_type"])) for row in rows]
        if relation == "上映年份":
            return f"上映于{items[0][0]}年"
        label = _MOVIE_RELATION_LABELS.get(relation, relation)
        return f"{label}：{_join(items)}"
    items = [(str(row["head"]), str(row["head_type"])) for row in rows]
    phrase = _SUBJECT_PHRASES.get(relation, f"相关{relation}")
    return f"{phrase}：{_join(items)}"


def _relation_order(edges: list[dict[str, Any]]) -> list[str]:
    """按边出现顺序取去重后的关系列表。"""
    return list(dict.fromkeys(e["relation"] for e in edges))


def _display_name(result: LinkResult) -> str:
    return f"《{result.entity}》" if result.entity_type == "电影" else result.entity


def _join(items: list[tuple[str, str]], cap: int = 20) -> str:
    """多值 tail 用顿号列举；电影加书名号，超出 cap 追加「等」。"""
    shown = items[:cap]
    text = "、".join(
        f"《{name}》" if ntype == "电影" else name for name, ntype in shown
    )
    return text + "等" if len(items) > cap else text


# ---- overview / recommend：子图 + FAISS 证据 → RAG 生成 ----


def _answer_rag(
    question: str,
    graph: nx.MultiDiGraph,
    linked: list[LinkResult],
    question_type: str,
    understanding: dict[str, Any],
    retriever: Retriever | None,
    llm: LLMClient | None,
    cfg: Config,
) -> dict[str, Any]:
    triples = _subgraph_triples(graph, linked)
    passages = _collect_passages(retriever, question, linked, cfg)

    reasons: list[str] = []
    hints: list[str] = []
    if retriever is None:
        # 契约 §6 v2.1：无 embedding key 时生成题直接 degraded（即使 llm 可用）
        reasons.append(
            f"未配置 {cfg.embedding.key_env}（SiliconFlow Embedding），"
            "向量检索与简介片段不可用"
        )
        hints.append(
            f"在仓库根目录 .env 配置 {cfg.embedding.key_env}"
            "（参考 .env.example）后重跑 python main.py index"
        )
    if llm is None:
        reasons.append(
            f"未配置 {cfg.llm.key_env}（silra），无法调用大模型生成自然语言回答"
        )
        hints.append(f"配置 {cfg.llm.key_env} 后可生成带 [n] 引用的完整回答")
    if reasons:
        understanding = {**understanding, "degraded_reason": "；".join(reasons)}
        return {
            "mode": "degraded",
            "answer": _degraded_answer(triples, passages, reasons, hints),
            "evidence": _evidence(understanding, triples, passages, _api_usage(llm)),
        }

    if not triples and not passages:
        return {
            "mode": "fallback",
            "answer": NO_EVIDENCE_REPLY,
            "evidence": _evidence(understanding, triples, passages, _api_usage(llm)),
        }

    assert llm is not None  # 走到这里说明 llm 与 retriever 都可用
    system = RAG_SYSTEM + (RAG_RECOMMEND_EXTRA if question_type == "recommend" else "")
    user = (
        f"问题：{question}\n\n证据：\n"
        + "\n".join(_serialize_evidence(triples, passages))
        + "\n\n请严格依据以上证据回答。"
    )
    text = llm.chat_text(system, user).strip()
    return {
        "mode": "rag",
        "answer": text or NO_EVIDENCE_REPLY,
        "evidence": _evidence(understanding, triples, passages, _api_usage(llm)),
    }


def _collect_passages(
    retriever: Retriever | None,
    question: str,
    linked: list[LinkResult],
    cfg: Config,
) -> list[dict[str, Any]]:
    """简介片段证据：检索池取宽（一次 encode 调用，多取不增加 API 成本），
    再把链接电影的自身简介块提到前排，见 _promote_linked_movie_passages。
    """
    if retriever is None:
        return []
    pool = retriever.search_synopsis(question, k=cfg.retriever.synopsis_top_k + 12)
    return _promote_linked_movie_passages(pool, linked, cfg.retriever.synopsis_top_k)


def _promote_linked_movie_passages(
    pool: list[dict[str, Any]],
    linked: list[LinkResult],
    top_k: int,
) -> list[dict[str, Any]]:
    """概述/生成题的简介证据保底：链接到的电影，其自身简介块优先入证据。

    「介绍一下《功夫》讲了什么故事」这类问题的向量检索容易被题材相近的
    其他影片简介挤出 top_k（都是功夫/江湖语境），因此从加宽的检索池里
    先取链接电影的块，再用其余高分块补足，保证「问哪部就带上哪部的简介」。
    """
    movie_names = {r.entity for r in linked if r.entity_type == "电影"}
    if not movie_names or not pool:
        return pool[:top_k]
    mine = [row for row in pool if row.get("title") in movie_names]
    others = [row for row in pool if row.get("title") not in movie_names]
    return (mine + others)[:top_k]


def _subgraph_triples(
    graph: nx.MultiDiGraph, linked: list[LinkResult], limit: int = 24
) -> list[dict[str, Any]]:
    """链接实体的 1-hop 子图三元组（先出边后入边，去重，限量）。"""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for result in linked:
        for u, v, data in list(graph.out_edges(result.entity, data=True)) + list(
            graph.in_edges(result.entity, data=True)
        ):
            key = (u, str(data.get("relation") or ""), v)
            if key in seen:
                continue
            seen.add(key)
            rows.append(_edge_row(u, v, data))
            if len(rows) >= limit:
                return rows
    return rows


def _serialize_evidence(
    triples: list[dict[str, Any]], passages: list[dict[str, Any]]
) -> list[str]:
    """证据编号：三元组 [1..T]、简介片段 [T+1..T+P]；evaluate.py 据此校验引用。"""
    lines: list[str] = []
    for index, row in enumerate(triples, 1):
        lines.append(
            f"[{index}] ({row['source'] or 'graph'}) "
            f"{row['head']}（{row['head_type']}）"
            f"-「{row['relation']}」→ {row['tail']}（{row['tail_type']}）"
        )
    offset = len(triples)
    for index, passage in enumerate(passages, offset + 1):
        lines.append(f"[{index}] 《{passage['title']}》简介片段：{passage['chunk']}")
    return lines


def _degraded_answer(
    triples: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    reasons: list[str],
    hints: list[str],
) -> str:
    lines = [f"【降级模式】{'；'.join(reasons)}。以下为本地检索到的证据："]
    numbered = _serialize_evidence(triples, passages)
    lines.extend(numbered if numbered else ["（未检索到相关证据）"])
    if hints:
        lines.append("提示：" + "；".join(hints) + "。")
    return "\n".join(lines)


def _api_usage(llm: LLMClient | None) -> dict[str, int] | None:
    """本次问答的 API 用量（llm.last_usage 的快照）；没有调用时为 None。"""
    return dict(llm.last_usage) if llm is not None else None


def _evidence(
    understanding: dict[str, Any],
    triples: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    api_usage: dict[str, int] | None,
) -> dict[str, Any]:
    """evidence 固定四键（契约 §6）。"""
    return {
        "understanding": understanding,
        "triples": triples,
        "passages": passages,
        "api_usage": api_usage,
    }
