"""详情入库（ingest）：data/raw/<subject_id>.json → 校验、过滤、
合并为 data/movies.jsonl。

单片原始记录由登录态浏览器采集落盘（备课期 Agent 驱动；学生可走 playwright
storage_state，见 --fetch）。字段名以实测为准：

    {"subject_id": "1291543", "url": "...", "search_title": "功夫‎ (2004)",
     "title": "功夫", "year": "2004", "info": "导演: ...\\n编剧: ...", "summary": "...",
     "rating": "8.9", "votes": "1372448", "collected_at": "..."}

info 是整块文本，逐行按「标签: 值」解析（兼容全角冒号）；「主演」行可能以
「/ 更多...」结尾，要去掉；多值用「/」分隔，但括号内的“/”（如日期备注）要保留，
因此按括号深度感知切分。

非电影过滤（研究计划 §3.1，逐条记录过滤日志到 data/ingest_report.json）：
  1. info 缺「类型」行，无法确认为电影；
  2. 类型含 综艺/真人秀/脱口秀/访谈/短片/纪录片；
  3. info 含「首播/集数/季数」行（电视剧集标志，可用配置关闭）；
  4. rating 与 votes 均为空（资料卡/非正片）；
  5. 导演/编剧/主演/配音均无“周星驰”（搜索误命中，可用配置关闭）。

输出 data/movies.jsonl（一行一部，随仓库分发）：
    {subject_id, url, title, year:int|null, rating:float|null, votes:int|null,
     directors, screenwriters, actors, genres, regions, languages,
     release_dates, runtimes, runtime_minutes:int|null, aka, imdb, summary,
     chow_roles}
  runtimes 保留豆瓣原样多版本列表（契约外附加键，与 chow_roles 同类）；
  runtime_minutes 是清洗后的主版本分钟数：备注含「中国大陆」的版本优先，
  否则取第一个可解析项（build_kg 据此只出一条「95分钟」格式的片长三元组）。

独立运行：
    python -m kgqa.collect_detail                 # 只合并已落盘的原始记录
    python -m kgqa.collect_detail --fetch         # 缺失的单片用 playwright 登录态补采
    python -m kgqa.collect_detail --storage-state s.json --no-headless
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import UTC, datetime
from typing import Any

from kgqa.config import Config, load_config

# “主演”行结尾的“/ 更多...”（豆瓣列表截断标志）
MORE_SUFFIX = re.compile(r"[／/]\s*更多[.…·]+\s*$")
# 信息行标签：标签较短且不含冒号（IMDb 值、Hong Kong: 108 分钟等出现在值里）
LINE_RE = re.compile(r"^([^:：]{1,14})[：:]\s*(.*)$")
# 电视剧集标志行
TV_LABELS = ("首播", "集数", "季数")
# 非电影类型关键词（命中即过滤）
NON_MOVIE_GENRES = ("综艺", "真人秀", "脱口秀", "访谈", "短片", "纪录片")
# chow_roles 的固定输出顺序
CHOW_ROLE_ORDER = ("导演", "编剧", "主演", "配音")
# 片长解析：「113分钟」「102 分钟」「100分钟(3D重映)」——前导 2-3 位数字，
# 「分钟」前允许空格，括号内为版本备注（地区/重映/修复版等）
RUNTIME_RE = re.compile(r"^(\d{2,3})\s*分钟\s*(?:[（(]([^（）()]{1,20})[）)])?\s*$")
# 豆瓣偶发的另一片长形态：「South Africa: 96 分钟」——地区前缀充当备注
RUNTIME_REGION_RE = re.compile(r"^([^:：]{1,20})[：:]\s*(\d{2,3})\s*分钟$")
# 多版本片长选主版本：备注含「中国大陆」的条目优先（境内公映版为准）
RUNTIME_MAIN_NOTE = "中国大陆"


def parse_info(info: str) -> dict[str, str]:
    """把 info 整块文本解析成 标签→值 字典（同标签取首次出现）。"""
    labels: dict[str, str] = {}
    for line in info.split("\n"):
        line = line.strip()
        match = LINE_RE.match(line)
        if match:
            labels.setdefault(match.group(1).strip(), match.group(2).strip())
    return labels


def split_slash(value: str) -> list[str]:
    """按“/”切分多值字段；括号内的“/”（如 2004-12-23(中国大陆/中国香港)）不切。"""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in value:
        if ch in "(（":
            depth += 1
        elif ch in ")）":
            depth = max(depth - 1, 0)
        if ch == "/" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def to_int(value: Any) -> int | None:
    """年份/评价人数转换；空串或非数字返回 None。"""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    return int(text) if text.isdigit() else None


def to_float(value: Any) -> float | None:
    """评分转换；空串或非法值返回 None。"""
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def list_field(labels: dict[str, str], label: str) -> list[str]:
    """多值字段：去掉“更多...”尾巴后按“/”切分、去重（保序）。"""
    raw = labels.get(label, "")
    values = split_slash(MORE_SUFFIX.sub("", raw))
    return list(dict.fromkeys(values))


def parse_runtime(item: str) -> tuple[int, str | None] | None:
    """单个片长字符串 → (分钟数, 版本备注|None)；解析失败返回 None。

    覆盖实测形态：`113分钟` / `102 分钟` / `100分钟(3D重映)` /
    `South Africa: 96 分钟`（地区前缀形态，前缀充当备注）。
    """
    match = RUNTIME_RE.match(item)
    if match:
        return int(match.group(1)), match.group(2) or None
    match = RUNTIME_REGION_RE.match(item)
    if match:
        return int(match.group(2)), match.group(1).strip() or None
    return None


def pick_runtime_minutes(items: list[str]) -> int | None:
    """多版本片长选主版本：备注含「中国大陆」的优先，否则取第一个可解析项。"""
    parsed = [p for p in (parse_runtime(item) for item in items) if p is not None]
    for minutes, note in parsed:
        if note and RUNTIME_MAIN_NOTE in note:
            return minutes
    return parsed[0][0] if parsed else None


def build_record(
    raw: dict[str, Any], cfg: Config
) -> tuple[dict[str, Any] | None, list[str]]:
    """单片原始记录 → movies.jsonl 一行；返回 (记录, 过滤原因列表)。"""
    reasons: list[str] = []
    info: str = raw.get("info") or ""
    labels = parse_info(info)

    genre_line = labels.get("类型")
    if not genre_line:
        reasons.append("info 缺「类型」行，无法确认为电影")
    else:
        genres = split_slash(genre_line)
        hit = [g for g in genres if g in NON_MOVIE_GENRES]
        if hit:
            reasons.append(f"类型含非电影分类：{'、'.join(hit)}")
    if cfg.douban.drop_tv_series and any(label in labels for label in TV_LABELS):
        reasons.append("info 含「首播/集数/季数」行，视为电视剧集")
    rating = to_float(raw.get("rating"))
    votes = to_int(raw.get("votes"))
    if rating is None and votes is None:
        reasons.append("rating 与 votes 均为空，疑似资料卡/非正片")
    creator_blob = "".join(
        labels.get(label, "") for label in ("导演", "编剧", "主演", "配音")
    )
    if cfg.douban.require_chow_credit and "周星驰" not in creator_blob:
        reasons.append("导演/编剧/主演/配音均无周星驰，按搜索误命中过滤")
    if reasons:
        return None, reasons

    # 标题形如“大话西游之大圣娶亲 西遊記大結局之仙履奇緣”：首个空格前是中文名，
    # 后面是原文名，原文名进 aka 供别名表使用
    full_title = str(raw.get("title") or "").strip()
    parts = full_title.split()
    title = parts[0] if parts else full_title
    aka = list_field(labels, "又名")
    if len(parts) > 1:
        aka = list(dict.fromkeys([*aka, " ".join(parts[1:])]))

    directors = list_field(labels, "导演")
    screenwriters = list_field(labels, "编剧")
    actors = list_field(labels, "主演")
    dubbers = list_field(labels, "配音")

    chow_roles: list[str] = []
    if "周星驰" in directors:
        chow_roles.append("导演")
    if "周星驰" in screenwriters:
        chow_roles.append("编剧")
    if "周星驰" in actors:
        chow_roles.append("主演")
    if "周星驰" in dubbers:
        chow_roles.append("配音")
    chow_roles.sort(key=CHOW_ROLE_ORDER.index)

    runtimes = list_field(labels, "片长")
    record = {
        "subject_id": str(raw["subject_id"]),
        "url": str(raw.get("url", "")),
        "title": title,
        "year": to_int(raw.get("year")),
        "rating": rating,
        "votes": votes,
        "directors": directors,
        "screenwriters": screenwriters,
        "actors": actors,
        "genres": list_field(labels, "类型"),
        "regions": list_field(labels, "制片国家/地区"),
        "languages": list_field(labels, "语言"),
        "release_dates": list_field(labels, "上映日期"),
        "runtimes": runtimes,
        "runtime_minutes": pick_runtime_minutes(runtimes),
        "aka": aka,
        "imdb": labels.get("IMDb", ""),
        "summary": str(raw.get("summary") or "").strip(),
        "chow_roles": chow_roles,
    }
    return record, []


def ingest(cfg: Config) -> dict[str, Any]:
    """遍历 subjects.json，合并全部已落盘的原始记录，写出 movies.jsonl 与过滤报告。"""
    subjects = load_subjects(cfg)
    records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for subject in subjects:
        raw_file = cfg.raw_dir / f"{subject['id']}.json"
        if not raw_file.exists() or raw_file.stat().st_size == 0:
            missing.append(
                {
                    "subject_id": subject["id"],
                    "title": subject["title"],
                    "url": subject["url"],
                }
            )
            continue
        raw = json.loads(raw_file.read_text(encoding="utf-8"))
        record, reasons = build_record(raw, cfg)
        if record is None:
            dropped.append(
                {
                    "subject_id": subject["id"],
                    "title": subject["title"],
                    "reasons": reasons,
                }
            )
        else:
            records.append(record)

    cfg.movies_file.parent.mkdir(parents=True, exist_ok=True)
    with cfg.movies_file.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "total_subjects": len(subjects),
        "kept": len(records),
        "dropped": dropped,
        "missing_raw": missing,
        "movies_file": str(cfg.movies_file),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    cfg.ingest_report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"入库完成：{cfg.movies_file}（{len(records)} 部）")
    print(
        f"过滤 {len(dropped)} 条、缺原始记录 {len(missing)} 条，"
        f"明细见 {cfg.ingest_report_file}"
    )
    for item in dropped:
        print(
            f"  丢弃 {item['title']}（{item['subject_id']}）："
            f"{'；'.join(item['reasons'])}"
        )
    if missing:
        print(
            "以下条目还没有 data/raw/<id>.json，需登录态浏览器采集（或 --fetch 补采）："
        )
        for item in missing[:10]:
            print(f"  {item['subject_id']}  {item['title']}")
        if len(missing) > 10:
            print(f"  … 共 {len(missing)} 条")
    return report


def load_subjects(cfg: Config) -> list[dict[str, Any]]:
    """读取搜索清单；没有则提示先跑 python -m kgqa.collect_list。"""
    if not cfg.subjects_file.exists():
        raise FileNotFoundError(
            f"没有找到 {cfg.subjects_file}，请先运行：python -m kgqa.collect_list"
        )
    subjects = json.loads(cfg.subjects_file.read_text(encoding="utf-8"))
    if not isinstance(subjects, list) or not subjects:
        raise ValueError(
            f"{cfg.subjects_file} 为空，请重新运行：python -m kgqa.collect_list"
        )
    return subjects


def fetch_missing(
    cfg: Config, storage_state: str | None = None, headless: bool = True
) -> int:
    """（可选）用 playwright 复用登录态，逐部补采缺失的详情页。

    需要先 `pip install playwright && playwright install chromium`；
    storage_state 由 `playwright 开有头浏览器手动登录后保存`（261101 作业任务一同款）。
    豆瓣详情页要求登录，若被拦截请用 --no-headless 重试。
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "未安装 playwright，无法补采。可跳过 --fetch 改为手动采集，或执行："
            "pip install playwright && playwright install chromium"
        ) from exc

    subjects = load_subjects(cfg)
    missing = [
        s
        for s in subjects
        if not (cfg.raw_dir / f"{s['id']}.json").exists()
        or (cfg.raw_dir / f"{s['id']}.json").stat().st_size == 0
    ]
    if not missing:
        print("所有条目均有原始记录，无需补采。")
        return 0

    low, high = cfg.douban.detail_sleep_sec
    print(
        f"待补采 {len(missing)} 部，限速 {low}-{high} 秒/部"
        f"（浏览器模式 headless={headless}）"
    )
    done = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=storage_state, locale="zh-CN")
        page = context.new_page()
        for subject in missing:
            print(f"  抓取 {subject['title']}（{subject['id']}）…")
            page.goto(subject["url"], wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector("#info", timeout=30000)
            title_el = page.locator("h1 span[property='v:itemreviewed']")
            year_el = page.locator("h1 .year")
            rating_el = page.locator("strong[property='v:average']")
            votes_el = page.locator("span[property='v:votes']")
            summary_els = page.locator("span[property='v:summary']")
            info = page.locator("#info").inner_text()
            summary = "\n".join(summary_els.all_inner_texts()).strip()
            raw = {
                "subject_id": subject["id"],
                "url": subject["url"],
                "search_title": subject["title"],
                "title": (
                    title_el.inner_text().strip()
                    if title_el.count()
                    else subject["title"]
                ),
                "year": (
                    year_el.inner_text().strip("() ")
                    if year_el.count()
                    else str(subject["year"])
                ),
                "info": info,
                "summary": summary,
                "rating": rating_el.inner_text().strip() if rating_el.count() else "",
                "votes": votes_el.inner_text().strip() if votes_el.count() else "",
                "collected_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            target = cfg.raw_dir / f"{subject['id']}.json"
            target.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            done += 1
            time.sleep(random.uniform(low, high))
        if storage_state:
            context.storage_state(path=storage_state)  # 顺带刷新登录态文件
        browser.close()
    print(f"补采完成 {done} 部；重新运行本命令即可合并入库。")
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description="豆瓣单片详情入库（raw → movies.jsonl）"
    )
    parser.add_argument("-c", "--config", default="configs/kg_qa.yaml", help="配置文件")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="缺失的单片用 playwright 登录态补采（需先安装 playwright）",
    )
    parser.add_argument(
        "--storage-state",
        default=None,
        help="playwright storage_state 文件路径",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="显示浏览器窗口（被风控拦截时使用）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.fetch:
        fetch_missing(
            cfg, storage_state=args.storage_state, headless=not args.no_headless
        )
    ingest(cfg)


if __name__ == "__main__":
    main()
