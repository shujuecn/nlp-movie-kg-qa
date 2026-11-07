"""确定性三元组与实体别名表构建（build-kg 子命令）。

读 data/movies.jsonl（ingest 产出），产出随仓库分发的 kg/ 文件：

  data/kg/triples.jsonl   每行 {head, head_type, relation, tail, tail_type, source}
                     relation 覆盖：导演/编剧/主演/类型/制片地区/语言/上映年份/片长/
                     评分/又名（「上映年份」从结构化字段取；「片长」取 ingest 清洗出
                     的主版本 runtime_minutes，每部至多一条，tail 形如 "95分钟"）；
                     source 均为 douban-info。
                     说明：属性型关系（年份/片长/评分/又名）的 tail 是字面量，
                     统一标注 tail_type=字面值（实体类型枚举之外唯一补充）。
  data/kg/aliases.jsonl   每行 {entity, entity_type, aliases, source}；别名来源：
                     电影又名与原文名（douban-aka）、周星驰本人英文名/昵称
                     （person-card，硬编码少量合理别名）。

语义三元组（饰演/题材）由 extract_llm.py 离线抽取，单独写 data/kg/triples_llm.jsonl，
与本文件互不影响；问答时两份三元组会合并加载。

独立运行：python -m kgqa.build_kg [-c configs/kg_qa.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kgqa.config import Config, load_config

PERSON_CHOW = "周星驰"
# 周星驰本人的硬编码别名（来源标注 person-card；如需扩充直接改这里）
CHOW_ALIASES = ["Stephen Chow", "星爷", "Chow Sing-chi"]

# 属性型关系的 tail 是字面量（年份、片长、评分、又名），统一用这个实体类型
LITERAL_TYPE = "字面值"


def movie_triples(movie: dict[str, Any]) -> list[dict[str, Any]]:
    """一部电影的结构化字段 → 三元组列表（source=douban-info）。"""
    head = movie["title"]
    triples: list[dict[str, Any]] = []

    def emit(relation: str, tails: list[Any], tail_type: str) -> None:
        for tail in tails:
            triples.append(
                {
                    "head": head,
                    "head_type": "电影",
                    "relation": relation,
                    "tail": str(tail),
                    "tail_type": tail_type,
                    "source": "douban-info",
                }
            )

    emit("导演", movie.get("directors") or [], "人物")
    emit("编剧", movie.get("screenwriters") or [], "人物")
    emit("主演", movie.get("actors") or [], "人物")
    emit("类型", movie.get("genres") or [], "类型")
    emit("制片地区", movie.get("regions") or [], "地区")
    emit("语言", movie.get("languages") or [], "语言")
    if movie.get("year") is not None:
        emit("上映年份", [movie["year"]], LITERAL_TYPE)
    # 片长只出一条清洗后的主版本（ingest 的 runtime_minutes，"95分钟"格式）；
    # 多版本 raw 列表与解析失败的（None）都不出三元组
    if movie.get("runtime_minutes") is not None:
        emit("片长", [f"{movie['runtime_minutes']}分钟"], LITERAL_TYPE)
    if movie.get("rating") is not None:
        emit("评分", [movie["rating"]], LITERAL_TYPE)
    emit("又名", movie.get("aka") or [], LITERAL_TYPE)
    return triples


def movie_alias_row(movie: dict[str, Any]) -> dict[str, Any] | None:
    """电影别名行：又名 + 原文名（ingest 已并入 aka），排除与片名重复的项。"""
    aliases = [
        a for a in dict.fromkeys(movie.get("aka") or []) if a and a != movie["title"]
    ]
    if not aliases:
        return None
    return {
        "entity": movie["title"],
        "entity_type": "电影",
        "aliases": aliases,
        "source": "douban-aka",
    }


def build(
    movies: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """全部电影 → (去重后的三元组列表, 别名表行列表)。"""
    seen: set[tuple[str, str, str]] = set()
    triples: list[dict[str, Any]] = []
    for movie in movies:
        for triple in movie_triples(movie):
            key = (triple["head"], triple["relation"], triple["tail"])
            if key not in seen:
                seen.add(key)
                triples.append(triple)

    alias_rows: list[dict[str, Any]] = []
    entities: dict[tuple[str, str], list[str]] = {}
    for triple in triples:
        if triple["relation"] == "又名":
            continue  # 属性值不是实体
        entities.setdefault((triple["tail"], triple["tail_type"]), [])
        entities.setdefault((triple["head"], triple["head_type"]), [])

    # 人物/类型/地区/语言先登记（默认无别名），电影用 aka 行覆盖
    for entity, entity_type in sorted(entities):
        if entity_type == "电影" or entity == PERSON_CHOW:
            continue  # 电影走 movie_alias_row；周星驰在循环后统一写入
        alias_rows.append(
            {
                "entity": entity,
                "entity_type": entity_type,
                "aliases": [],
                "source": "douban-info",
            }
        )

    for movie in movies:
        row = movie_alias_row(movie)
        if row:
            alias_rows.append(row)

    # 周星驰本人：英文名与昵称，来源 person-card
    alias_rows.append(
        {
            "entity": PERSON_CHOW,
            "entity_type": "人物",
            "aliases": list(CHOW_ALIASES),
            "source": "person-card",
        }
    )
    return triples, alias_rows


def run(cfg: Config) -> tuple[int, int]:
    """供 main.py build-kg 调用；返回 (三元组数, 别名行数)。"""
    if not cfg.movies_file.exists():
        raise FileNotFoundError(
            f"没有找到 {cfg.movies_file}，请先运行：python main.py ingest"
        )
    movies = [
        json.loads(line)
        for line in cfg.movies_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not movies:
        raise ValueError(
            f"{cfg.movies_file} 为空，请先完成采集与入库（main.py ingest）"
        )

    triples, alias_rows = build(movies)
    cfg.kg_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(cfg.triples_file, triples)
    _write_jsonl(cfg.aliases_file, alias_rows)

    n_movies = len(movies)
    n_persons = sum(1 for row in alias_rows if row["entity_type"] == "人物")
    print(f"读入电影 {n_movies} 部")
    print(f"三元组 {len(triples)} 条 → {cfg.triples_file}")
    print(f"实体别名 {len(alias_rows)} 行（人物 {n_persons} 行）→ {cfg.aliases_file}")
    print("注意：data/kg/ 只含结构化短句；剧情简介全文在 data/movies.jsonl。")
    return len(triples), len(alias_rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="结构化字段 → 确定性三元组 + 实体别名表"
    )
    parser.add_argument("-c", "--config", default="configs/kg_qa.yaml", help="配置文件")
    args = parser.parse_args()

    run(load_config(args.config))


if __name__ == "__main__":
    main()
