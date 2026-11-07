"""离线语义三元组抽取（silra API，JSON mode）。

读 data/movies.jsonl，把每部影片的剧情简介交给大模型，抽取图谱增量：
「角色-饰演者」与「题材主题」类语义三元组（导演/主演/评分等确定性字段
已由 build_kg.py 落图谱，这里不重复抽取）。

输出 kg/triples_llm.jsonl，每行在契约 schema 基础上加 subject_id（重跑去重用）：
    {subject_id, head, head_type, relation, tail, tail_type, source}
relation 实际只会是 饰演（head=角色名/head_type=角色，tail=演员名/tail_type=人物）
或 题材（tail=主题词/tail_type=主题）；source 固定 llm-offline。
写文件采用「读旧文件 → 剔除本次重跑的 subject_id → 接上新记录」，每抽完一部
落盘一次：重跑同一部会覆盖它的旧记录，其他影片的历史结果不受影响。

独立运行（需先配置 SILRA_API_KEY，根目录 .env 会自动加载）：
    python extract_llm.py --subject 1291543        抽取单部
    python extract_llm.py --all                    全量抽取
    python extract_llm.py --all --limit 5          只抽前 5 部（试跑）
可选 --out 覆盖输出文件（默认配置里的 kg.triples_llm_file）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kgqa.config import ENTITY_TYPES, RELATIONS, Config, load_config
from kgqa.llm_client import LLMClient, create_client, has_api_key, load_env_file

__all__ = ["extract_subject"]

# 契约 §2 的取值范围拼进提示词；「字面值」是属性型 tail 的内部标注，不进提示词
_TYPE_RANGE = "、".join(t for t in ENTITY_TYPES if t != "字面值")
_RELATION_RANGE = "、".join(RELATIONS)

EXTRACT_SYSTEM = f"""\
你是电影知识图谱构建助手，任务是从影片剧情简介中抽取语义三元组。

输出格式：只输出一个 JSON 对象，形如
{{"triples": [{{"head": "...", "head_type": "...", "relation": "...",
  "tail": "...", "tail_type": "..."}}]}}

字段取值范围（契约 §2）：
- head_type / tail_type 只能是：{_TYPE_RANGE}
- relation 只能是：{_RELATION_RANGE}

抽取重点（周星驰电影领域）：
1. 角色-饰演者：relation=饰演，head=角色名（head_type=角色），
   tail=演员名（tail_type=人物）。
   例：{{"head": "紫霞仙子", "head_type": "角色", "relation": "饰演",
        "tail": "朱茵", "tail_type": "人物"}}
2. 题材主题：relation=题材，tail=主题词（tail_type=主题）。
   例：{{"head": "功夫", "head_type": "电影", "relation": "题材",
        "tail": "功夫", "tail_type": "主题"}}

要求：
- 每部影片 3-6 条，优先「角色-饰演者」对（角色名与演员名都要在简介或主演列表中出现过）；
- 只抽简介中明确出现的信息，不要编造片中没有的角色或演员；
- 导演/主演/评分等结构化字段已另行入库，不要重复抽取。
"""


def extract_subject(llm: LLMClient, movie: dict[str, Any]) -> list[dict[str, Any]]:
    """单部影片：简介 → 语义三元组（source=llm-offline）；无简介或抽不出返回 []。"""
    summary = str(movie.get("summary") or "").strip()
    if not summary:
        print(f"  跳过（无简介）：{movie.get('title')}")
        return []
    actors = "、".join(str(a) for a in (movie.get("actors") or []))
    user = (
        f"影片：《{movie.get('title')}》\n"
        f"主演：{actors or '（未知）'}\n\n"
        f"剧情简介：\n{summary}"
    )
    data = llm.chat_json(EXTRACT_SYSTEM, user)
    return _normalize(data, movie)


def _normalize(data: dict[str, Any], movie: dict[str, Any]) -> list[dict[str, Any]]:
    """校验模型输出：只保留 饰演/题材 且类型合法的行，补 subject_id 与 source。"""
    rows = data.get("triples") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        print("  警告：模型输出缺少 triples 数组，按 0 条处理")
        return []
    subject_id = str(movie.get("subject_id") or "")
    title = str(movie.get("title") or "")
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        head = str(row.get("head") or "").strip()
        head_type = str(row.get("head_type") or "").strip()
        relation = str(row.get("relation") or "").strip()
        tail = str(row.get("tail") or "").strip()
        tail_type = str(row.get("tail_type") or "").strip()
        if not head or not tail:
            continue
        if relation not in ("饰演", "题材"):
            print(f"  跳过非语义关系（确定性三元组已覆盖）：{head}-{relation}->{tail}")
            continue
        if relation == "饰演" and (head_type != "角色" or tail_type != "人物"):
            print(
                f"  跳过类型不符的饰演三元组："
                f"{head}({head_type})-{relation}->{tail}({tail_type})"
            )
            continue
        if relation == "题材" and tail_type != "主题":
            print(f"  跳过类型不符的题材三元组：{head}-{relation}->{tail}({tail_type})")
            continue
        type_valid = head_type in ENTITY_TYPES and tail_type in ENTITY_TYPES
        if "字面值" in (head_type, tail_type) or not type_valid:
            print(
                f"  跳过实体类型越界的三元组："
                f"{head}({head_type})-{relation}->{tail}({tail_type})"
            )
            continue
        key = (head, relation, tail)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "subject_id": subject_id,
                "head": head,
                "head_type": head_type,
                "relation": relation,
                "tail": tail,
                "tail_type": tail_type,
                "source": "llm-offline",
            }
        )
    if not results:
        print(f"  警告：《{title}》没有抽到合法三元组")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="简介 → 语义三元组离线抽取（silra API）"
    )
    parser.add_argument("-c", "--config", default="configs/kg_qa.yaml", help="配置文件")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject", default=None, help="只抽取指定 subject_id 的影片")
    group.add_argument("--all", action="store_true", help="抽取 movies.jsonl 全部影片")
    parser.add_argument(
        "--limit", type=int, default=None, help="配合 --all 只抽前 N 部"
    )
    parser.add_argument(
        "--out", default=None, help="输出文件（默认配置 kg.triples_llm_file）"
    )
    args = parser.parse_args()

    try:
        _run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from None


def _run(args: argparse.Namespace) -> None:
    load_env_file()
    cfg = load_config(args.config)
    if not has_api_key(cfg.llm):
        raise SystemExit(
            f"错误：环境变量 {cfg.llm.key_env} 未配置，无法调用 silra API。"
            "请在仓库根目录 .env 里填入 key（参考 .env.example）。"
        )

    movies = _load_movies(cfg)
    if args.subject:
        movies = [m for m in movies if str(m.get("subject_id")) == args.subject]
        if not movies:
            raise ValueError(
                f"{cfg.movies_file} 里没有 subject_id={args.subject} 的影片"
            )
    elif args.limit is not None:
        movies = movies[: max(args.limit, 0)]

    out_path = Path(args.out) if args.out else cfg.triples_llm_file
    llm = create_client(cfg.llm)
    _run_extraction(llm, movies, out_path)


def _run_extraction(
    llm: LLMClient, movies: list[dict[str, Any]], out_path: Path
) -> None:
    """逐部抽取；每抽完一部把「旧记录剔除本次已重跑的 subject_id + 新记录」落盘一次。"""
    old_records = _read_jsonl(out_path) if out_path.exists() else []
    done: dict[str, list[dict[str, Any]]] = {}  # 本次已重跑的 subject_id → 新记录
    total_triples = 0
    print(f"待抽取影片 {len(movies)} 部 → {out_path}")
    for index, movie in enumerate(movies, 1):
        subject_id = str(movie.get("subject_id") or "")
        title = str(movie.get("title") or "")
        print(f"[{index}/{len(movies)}] 《{title}》（{subject_id}）抽取中……")
        try:
            triples = extract_subject(llm, movie)
        except (RuntimeError, ValueError) as exc:
            print(f"  抽取失败，跳过：{exc}")
            continue
        done[subject_id] = triples
        total_triples += len(triples)
        print(f"  得到 {len(triples)} 条三元组；本次调用 token 用量：{llm.last_usage}")
        _write_merged(out_path, old_records, done)

    print(
        f"完成：成功 {len(done)}/{len(movies)} 部，新增/覆盖三元组 {total_triples} 条"
    )
    print(f"输出文件：{out_path}")
    print(f"累计 token 用量：{json.dumps(llm.usage, ensure_ascii=False)}")


def _write_merged(
    out_path: Path,
    old_records: list[dict[str, Any]],
    done: dict[str, list[dict[str, Any]]],
) -> None:
    """旧记录剔除本次重跑的 subject_id，再接上新记录，整体覆盖写。"""
    kept = [row for row in old_records if str(row.get("subject_id") or "") not in done]
    rows = [row for triples in done.values() for row in triples]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in [*kept, *rows]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_movies(cfg: Config) -> list[dict[str, Any]]:
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
    return movies


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
