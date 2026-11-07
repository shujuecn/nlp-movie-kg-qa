"""问题理解模块：LLM zero-shot 结构化理解 / 规则降级两条路线。

从 kg_qa.py 拆出（v2.2 模块化）。教学对照点（承接 261101）：
上一课用 bge+SVM 分类器做意图识别，本课用 LLM 的 JSON mode 做 zero-shot
结构化理解；规则降级路线保证无 key 时系统仍可用。

输出契约（接口契约 §6）：
    {"entity_mentions": [...], "question_type": "fact|overview|recommend|out_of_scope",
     "target_relations": [...]}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from kgqa.config import QUESTION_TYPES, RELATIONS, Config
from kgqa.kg_linker import load_aliases, row_names
from kgqa.llm_client import LLMClient

__all__ = ["relations_from_question", "understand_question"]

# 规则降级的 fact 启发式（契约 §6）：长度 < 18 且含这些词 → fact。
# 在契约给出的 是谁|哪部|几年|评分 基础上补了周星驰领域常见问法（谁演/哪年/多少/哪些）
_FACT_HINTS = ("是谁", "哪部", "几年", "评分", "谁演", "哪年", "多少", "哪些")

# 问题关键词 → 目标关系（按优先级排列；规则理解与 fact 兜底选关系共用）
_RELATION_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("谁演", "饰演"),
    ("演的是谁", "饰演"),
    ("饰演", "饰演"),
    ("扮演", "饰演"),
    ("导演", "导演"),
    ("执导", "导演"),
    ("编剧", "编剧"),
    ("主演", "主演"),
    ("出演", "主演"),
    ("参演", "主演"),
    ("演过", "主演"),
    ("配音", "配音"),
    ("评分", "评分"),
    ("几分", "评分"),
    ("多少分", "评分"),
    ("上映", "上映年份"),
    ("哪一年", "上映年份"),
    ("哪年", "上映年份"),
    ("几年", "上映年份"),
    ("什么时候", "上映年份"),
    ("类型", "类型"),
    ("题材", "题材"),
    ("主题", "题材"),
    ("又名", "又名"),
    ("又叫", "又名"),
    ("还叫", "又名"),
    ("别名", "又名"),
    ("片长", "片长"),
    ("多长", "片长"),
    ("国家", "制片地区"),
    ("地区", "制片地区"),
    ("语言", "语言"),
)

# 规则降级的实体候选缓存（键：别名表 + llm 三元组文件路径；进程内有效）
_RULE_NAME_CACHE: dict[str, list[str]] = {}

_QUOTE_PATTERNS = (
    re.compile(r"《([^《》]{1,30})》"),
    re.compile(r"[「『“\"]([^」』”\"]{1,30})[」』”\"]"),
)

UNDERSTAND_SYSTEM = """\
你是周星驰电影知识图谱问答系统的「问题理解」模块。分析用户问题，只输出一个 JSON 对象：
{"entity_mentions": [], "question_type": "fact", "target_relations": []}

字段要求：
- entity_mentions：问题中出现的实体原文（电影名/人物名/角色名/类型名等），
  保持问题中的写法；
- question_type 只能是 fact（问具体事实）/ overview（介绍、剧情、评价类概述）/
  recommend（求推荐）/ out_of_scope（与周星驰电影无关的问题）；
- target_relations 只能取：导演、编剧、主演、配音、类型、制片地区、语言、
  上映年份、片长、评分、又名、饰演、题材；判断不出时给 []。

例子（每行一个 JSON 对象）：
「功夫是谁导演的」→
{"entity_mentions": ["功夫"], "question_type": "fact", "target_relations": ["导演"]}
「紫霞仙子是谁演的」→
{"entity_mentions": ["紫霞仙子"], "question_type": "fact", "target_relations": ["饰演"]}
「大话西游之大圣娶亲是哪年上映的」→
{"entity_mentions": ["大话西游之大圣娶亲"], "question_type": "fact",
 "target_relations": ["上映年份"]}
「周星驰和吴孟达合作过哪些电影」→
{"entity_mentions": ["周星驰", "吴孟达"], "question_type": "fact",
 "target_relations": ["主演", "导演", "编剧"]}
「介绍一下食神」→
{"entity_mentions": ["食神"], "question_type": "overview", "target_relations": []}
「推荐几部周星驰的喜剧电影」→
{"entity_mentions": ["周星驰", "喜剧"], "question_type": "recommend",
 "target_relations": ["类型"]}
「帮我写一首诗」→
{"entity_mentions": [], "question_type": "out_of_scope", "target_relations": []}

只输出 JSON，不要输出解释文字。
"""


def understand_question(
    llm: LLMClient | None, cfg: Config, question: str
) -> dict[str, Any]:
    """问题理解：返回 {"entity_mentions": [...], "question_type": ...,
    "target_relations": [...]}（契约 §6）。

    llm 不为 None 走 API JSON 模式（zero-shot 结构化理解）；
    llm 为 None 走规则降级：书名号/引号内与候选名命中的词为实体提及，
    问题类型按长度与疑问词启发式，目标关系按关键词映射。
    """
    if llm is not None:
        return _understand_with_llm(llm, question)
    return _understand_by_rules(cfg, question)


def relations_from_question(question: str) -> list[str]:
    """从问题文本按关键词映射目标关系（规则理解与 fact 兜底共用）。"""
    relations: list[str] = []
    for keyword, relation in _RELATION_KEYWORDS:
        if keyword in question and relation not in relations:
            relations.append(relation)
    return relations


def _understand_with_llm(llm: LLMClient, question: str) -> dict[str, Any]:
    data = llm.chat_json(UNDERSTAND_SYSTEM, f"用户问题：{question}")
    mentions = _dedupe(
        [str(m).strip() for m in (data.get("entity_mentions") or []) if str(m).strip()]
    )
    question_type = str(data.get("question_type") or "").strip()
    if question_type not in QUESTION_TYPES:
        question_type = "overview"  # 未知类型按概述兜底，走证据生成
    relations = [r for r in (data.get("target_relations") or []) if r in RELATIONS]
    return {
        "entity_mentions": mentions,
        "question_type": question_type,
        "target_relations": _dedupe(relations),
    }


def _understand_by_rules(cfg: Config, question: str) -> dict[str, Any]:
    if "推荐" in question:
        question_type = "recommend"
    elif len(question) < 18 and any(hint in question for hint in _FACT_HINTS):
        question_type = "fact"
    else:
        question_type = "overview"
    return {
        "entity_mentions": _rule_mentions(cfg, question),
        "question_type": question_type,
        "target_relations": relations_from_question(question),
    }


def _rule_mentions(cfg: Config, question: str) -> list[str]:
    """规则降级的实体提及：书名号/引号内容 + 候选名命中。

    候选按「先长后短」过滤（同位置长名优先，被更长命中覆盖的短名跳过），
    最终按提及在问题中的出现位置排序。
    """
    found: list[tuple[int, str]] = []
    for pattern in _QUOTE_PATTERNS:
        for match in pattern.finditer(question):
            text = match.group(1).strip()
            if text:
                found.append((match.start(1), text))
    for name in _rule_name_candidates(cfg):
        pos = question.find(name)
        if pos < 0:
            continue
        if any(name in text or text in name for _, text in found):
            continue
        found.append((pos, name))
        if len(found) >= 8:
            break
    found.sort(key=lambda pair: pair[0])
    return [text for _, text in found]


def _rule_name_candidates(cfg: Config) -> list[str]:
    """规则降级的实体候选名：别名表实体名/别名 + llm 三元组的角色/主题名（带缓存）。

    角色/主题实体不在别名表里（build_kg 不认识 llm 三元组），必须从
    data/kg/triples_llm.jsonl 补齐，「紫霞仙子是谁演的」这类降级题才能抽出提及。
    """
    key = f"{cfg.aliases_file}|{cfg.triples_llm_file}"
    if key not in _RULE_NAME_CACHE:
        names: list[str] = []
        try:
            for row in load_aliases(cfg.aliases_file):
                names.extend(row_names(row))
        except FileNotFoundError:
            pass  # 没有别名表时仅靠书名号/引号
        llm_file = Path(cfg.triples_llm_file)
        if llm_file.exists():
            for line in llm_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for node, node_type in (
                    (row.get("head"), row.get("head_type")),
                    (row.get("tail"), row.get("tail_type")),
                ):
                    if node_type in ("角色", "主题") and str(node or "").strip():
                        names.append(str(node).strip())
        # 先保序去重再排序：等长候选名保持别名表登记次序，结果不受 PYTHONHASHSEED 影响
        _RULE_NAME_CACHE[key] = sorted(_dedupe(names), key=len, reverse=True)
    return _RULE_NAME_CACHE[key]


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
