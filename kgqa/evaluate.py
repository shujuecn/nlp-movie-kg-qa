"""评测：读 data/eval/qa_eval.jsonl，逐题调 kgqa.qa.answering.answer，
输出三项指标与分题型摘要。

指标（契约 §8）：
- 链接命中率：抽出了实体提及（entity_mentions 非空）的题目里，至少成功链接
  一个实体的比例；
- 答案关键词命中率：gold_keywords 任一出现在答案文本中即算命中（英文不区分大小写）；
- 生成题引用有效率：mode=rag 的题目里，答案含 [n] 引用且每个 n 都落在证据编号
  范围内的比例（三元组 [1..T] + 简介片段 [T+1..T+P]，
  编号规则见 kgqa.qa.answering._serialize_evidence）。

无 SILRA key 时生成题按 degraded 评测（引用有效率口径仍只统计 mode=rag）；
无 SILICONFLOW key 时 retriever=None，生成题没有简介片段证据。
评测集缺失时中文报错（CLI 捕获后以 SystemExit 退出，不崩栈）。

独立运行：python -m kgqa.evaluate [-c configs/kg_qa.yaml]
报告写 data/eval_report.json（总分 + 分题型 + 逐题明细）。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Any

from kgqa.config import Config, load_config
from kgqa.embed_client import has_embed_key
from kgqa.graph import load_graph
from kgqa.kg_linker import load_aliases
from kgqa.llm_client import LLMClient, create_client, has_api_key, load_env_file
from kgqa.qa.answering import answer
from kgqa.qa.retriever import Retriever

_CITATION_RE = re.compile(r"\[(\d+)\]")


def run(cfg: Config) -> dict[str, Any]:
    """评测主流程：构建/加载组件 → 逐题作答 → 统计 → 打印摘要并写报告。"""
    graph = load_graph(cfg.kg_dir)
    aliases = load_aliases(cfg.aliases_file)
    retriever = _build_retriever(cfg)
    llm = _build_llm(cfg)

    items = _load_eval_items(cfg)
    details: list[dict[str, Any]] = []
    mode_counter: Counter[str] = Counter()
    print(
        f"\n开始评测：{len(items)} 道题（llm={'API' if llm else '规则降级'}，"
        f"retriever={'FAISS' if retriever else '无向量检索'}）"
    )
    for index, item in enumerate(items, 1):
        question = str(item.get("question") or "")
        try:
            result = answer(
                question,
                graph=graph,
                aliases=aliases,
                retriever=retriever,
                llm=llm,
                cfg=cfg,
            )
        except RuntimeError as exc:  # 单题失败不拖垮整个评测（API 网络错误等）
            print(f"[{index}/{len(items)}] 作答失败：{exc}")
            result = {
                "mode": "fallback",
                "answer": f"（本题作答失败：{exc}）",
                "evidence": {
                    "understanding": {},
                    "triples": [],
                    "passages": [],
                    "api_usage": None,
                },
            }
        detail = _judge(item, result)
        details.append(detail)
        mode_counter[str(result.get("mode"))] += 1
        verdict = "命中" if detail["keyword_hit"] else "未命中"
        print(
            f"[{index}/{len(items)}] {detail['type']} mode={result.get('mode')} "
            f"关键词{verdict}｜{question}"
        )

    report = _summarize(details, mode_counter)
    _print_summary(report)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg.eval_report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"评测报告已写出：{cfg.eval_report_file}")
    return report


# ---- 组件构建 ----


def _build_retriever(cfg: Config) -> Retriever | None:
    """有 SILICONFLOW key 才构建 Retriever（索引缺失/过期会在其内部自动重建）。"""
    if not has_embed_key(cfg.embedding):
        print(
            f"未配置 {cfg.embedding.key_env}，向量检索不可用："
            "生成题将按 degraded 评测（无简介片段证据）。"
        )
        return None
    try:
        return Retriever(cfg, cfg.embedding)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"警告：Retriever 初始化失败（{exc}），生成题将没有简介片段证据。")
        return None


def _build_llm(cfg: Config) -> LLMClient | None:
    """有 SILRA key 构建客户端；无 key 走规则降级（fact 题不受影响）。"""
    if not has_api_key(cfg.llm):
        print(
            f"未配置 {cfg.llm.key_env}，评测以降级模式运行："
            "生成题只统计检索证据，不计引用有效率。"
        )
        return None
    try:
        return create_client(cfg.llm)
    except RuntimeError as exc:  # LLMError（含 key 校验）都是 RuntimeError 子类
        print(f"警告：LLM 客户端初始化失败（{exc}），评测以降级模式运行。")
        return None


def _load_eval_items(cfg: Config) -> list[dict[str, Any]]:
    path = cfg.eval_file
    if not path.exists():
        raise FileNotFoundError(
            f"没有找到评测集 {path}。请先准备 data/eval/qa_eval.jsonl，"
            "每行一道题：{qid, question, type, gold_keywords: []}"
            "（schema 见 docs/接口契约.md §2）。"
        )
    items = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        raise ValueError(f"评测集 {path} 为空")
    return items


# ---- 单题金标比对与汇总 ----


def _judge(item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """单题金标比对：链接命中 / 关键词命中 / 引用有效。"""
    evidence = result.get("evidence") or {}
    understanding = evidence.get("understanding") or {}
    linked = understanding.get("linked") or []
    answer_text = str(result.get("answer") or "")
    gold_keywords = [str(k) for k in (item.get("gold_keywords") or []) if str(k)]
    lowered = answer_text.casefold()
    keyword_hit = any(k.casefold() in lowered for k in gold_keywords)

    citations: list[int] = []
    citation_valid: bool | None = None
    if result.get("mode") == "rag":
        citations = [int(n) for n in _CITATION_RE.findall(answer_text)]
        span = len(evidence.get("triples") or []) + len(evidence.get("passages") or [])
        citation_valid = bool(citations) and all(1 <= n <= span for n in citations)

    return {
        "qid": item.get("qid"),
        "type": str(item.get("type") or "unknown"),
        "question": item.get("question"),
        "gold_keywords": gold_keywords,
        "gold_relation": item.get("gold_relation"),
        "mode": result.get("mode"),
        "answer": answer_text,
        "mention_count": len(understanding.get("entity_mentions") or []),
        "link_hit": bool(linked),
        "linked_entities": [row.get("entity") for row in linked],
        "keyword_hit": keyword_hit,
        "citations": citations,
        "citation_valid": citation_valid,
    }


def _summarize(
    details: list[dict[str, Any]], mode_counter: Counter[str]
) -> dict[str, Any]:
    """三项指标 + 分题型汇总（口径写进 definitions，方便课堂讲解）。"""
    total = len(details)
    link_candidates = [d for d in details if d["mention_count"] > 0]
    link_hits = sum(1 for d in link_candidates if d["link_hit"])
    keyword_hits = sum(1 for d in details if d["keyword_hit"])
    rag_details = [d for d in details if d["mode"] == "rag"]
    citation_valid = sum(1 for d in rag_details if d["citation_valid"])

    by_type: dict[str, dict[str, Any]] = {}
    for d in details:
        bucket = by_type.setdefault(
            d["type"],
            {
                "n": 0,
                "link_candidates": 0,
                "link_hits": 0,
                "keyword_hits": 0,
                "rag_n": 0,
                "citation_valid": 0,
            },
        )
        bucket["n"] += 1
        if d["mention_count"] > 0:
            bucket["link_candidates"] += 1
            if d["link_hit"]:
                bucket["link_hits"] += 1
        if d["keyword_hit"]:
            bucket["keyword_hits"] += 1
        if d["mode"] == "rag":
            bucket["rag_n"] += 1
            if d["citation_valid"]:
                bucket["citation_valid"] += 1

    return {
        "total": total,
        "definitions": {
            "link_hit_rate": "抽出了实体提及的题目中，至少成功链接一个实体的比例",
            "keyword_hit_rate": "gold_keywords 任一出现在答案文本中的题目比例",
            "citation_valid_rate": (
                "mode=rag 的题目中，答案含 [n] 且 n 落在证据编号范围的比例"
            ),
        },
        "mode_counts": dict(mode_counter),
        "metrics": {
            "link_hit_rate": (
                link_hits / len(link_candidates) if link_candidates else 0.0
            ),
            "link_hits": link_hits,
            "link_candidates": len(link_candidates),
            "keyword_hit_rate": keyword_hits / total if total else 0.0,
            "keyword_hits": keyword_hits,
            "citation_valid_rate": (
                citation_valid / len(rag_details) if rag_details else None
            ),
            "citation_valid": citation_valid,
            "rag_questions": len(rag_details),
        },
        "by_type": by_type,
        "details": details,
    }


def _print_summary(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    modes = (
        "、".join(f"{mode}={n}" for mode, n in sorted(report["mode_counts"].items()))
        or "-"
    )
    print("\n========== 评测摘要 ==========")
    print(f"题目总数：{report['total']}｜模式分布：{modes}")
    print(
        f"链接命中率：{metrics['link_hit_rate']:.4f}"
        f"（{metrics['link_hits']}/{metrics['link_candidates']} 道有实体提及的题）"
    )
    print(
        f"答案关键词命中率：{metrics['keyword_hit_rate']:.4f}"
        f"（{metrics['keyword_hits']}/{report['total']}）"
    )
    if metrics["rag_questions"]:
        print(
            f"生成题引用有效率：{metrics['citation_valid_rate']:.4f}"
            f"（{metrics['citation_valid']}/{metrics['rag_questions']}，"
            "仅统计 mode=rag）"
        )
    else:
        print(
            "生成题引用有效率：本题集没有 mode=rag 的题目可统计"
            "（降级模式生成题不计入）。"
        )

    print("\n题型 | 题数 | 链接命中 | 关键词命中 | 引用有效")
    for name, bucket in report["by_type"].items():
        link_text = (
            f"{bucket['link_hits']}/{bucket['link_candidates']}"
            if bucket["link_candidates"]
            else "-"
        )
        rag_text = (
            f"{bucket['citation_valid']}/{bucket['rag_n']}" if bucket["rag_n"] else "-"
        )
        print(
            f"{name} | {bucket['n']} | {link_text} | "
            f"{bucket['keyword_hits']}/{bucket['n']} | {rag_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="问答系统评测（链接命中 / 关键词命中 / 引用有效）"
    )
    parser.add_argument("-c", "--config", default="configs/kg_qa.yaml", help="配置文件")
    args = parser.parse_args()
    try:
        load_env_file()
        run(load_config(args.config))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from None


if __name__ == "__main__":
    main()
