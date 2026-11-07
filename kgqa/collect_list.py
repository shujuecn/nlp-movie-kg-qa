"""豆瓣搜索列表页采集（urllib 标准库，免登录、免 requests）。

抓取 search.douban.com/movie/subject_search 的搜索结果：
源码里的 ``window.__DATA__`` 是明文 JSON，用 ``json.JSONDecoder().raw_decode`` 解析，
含 count / start / total / items，每页 15 条，按 ``start`` 步进翻页。

过滤规则（研究计划 §3.1）：
  1. 标题没有 ``(年份)`` 后缀的是人物卡（如“周星驰 Stephen Chow”），丢弃；
  2. 年份大于 douban.max_year（默认 2026）的未上映条目，丢弃；
  3. 按 subject id 去重。

输出（data/raw/ 下，均 gitignore）：
  subjects.json         过滤后的条目数组 [{id, title, year, url, abstract, abstract_2}]
  search_items_raw.json 全量原始 item 数组（未过滤，留作核对）

断点续爬：subjects.json 已存在且非空时跳过采集（--force 重爬）。

独立运行：
    python collect_list.py                    # 用 configs/kg_qa.yaml 的 douban 配置
    python collect_list.py --max-pages 2      # 只翻两页，快速验证
    python collect_list.py --force            # 忽略已有结果重爬
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from typing import Any

from kgqa.config import Config, load_config

SEARCH_URL = "https://search.douban.com/movie/subject_search"
PAGE_SIZE = 15  # 每页条数，与豆瓣返回的 payload["count"] 一致
DATA_MARKER = "window.__DATA__"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
YEAR_IN_TITLE = re.compile(r"\((\d{4})\)\s*$")  # 标题结尾的“(年份)”后缀


def parse_page_json(html: str) -> dict[str, Any]:
    """从页面源码定位 window.__DATA__ 并 raw_decode 出明文 JSON。"""
    pos = html.find(DATA_MARKER)
    if pos < 0:
        raise ValueError(
            "页面里没有找到 window.__DATA__，豆瓣可能改版或触发了风控，稍后重试"
        )
    brace = html.find("{", pos)
    if brace < 0:
        raise ValueError("window.__DATA__ 后没有找到 JSON 内容")
    payload, _ = json.JSONDecoder().raw_decode(html[brace:])
    if not isinstance(payload, dict):
        raise ValueError("window.__DATA__ 不是键值表，豆瓣可能改版")
    return payload


def fetch_page(search_text: str, start: int, timeout: float = 30.0) -> dict[str, Any]:
    """抓取一页搜索结果，返回解析后的 payload（count/start/total/items）。"""
    query = urllib.parse.urlencode({"search_text": search_text, "start": start})
    request = urllib.request.Request(
        f"{SEARCH_URL}?{query}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        html = response.read().decode("utf-8")
    payload = parse_page_json(html)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("window.__DATA__ 里没有 items 列表，豆瓣可能改版")
    return payload


def extract_year(item: dict[str, Any]) -> int | None:
    """人物卡标题没有 (年份) 后缀，返回 None；有则返回年份数字。"""
    match = YEAR_IN_TITLE.search(str(item.get("title", "")))
    return int(match.group(1)) if match else None


def to_subject(item: dict[str, Any], year: int) -> dict[str, Any]:
    """搜索 item → subjects.json 元素（只留后续采集与别名要用的字段）。"""
    return {
        "id": str(item["id"]),
        "title": str(item.get("title", "")).strip(),
        "year": year,
        "url": str(item.get("url", "")),
        "abstract": str(item.get("abstract", "")).strip(),
        "abstract_2": str(item.get("abstract_2", "")).strip(),
    }


def collect(
    cfg: Config, max_pages: int | None = None, force: bool = False
) -> list[dict[str, Any]]:
    """翻页抓取并过滤，写出 subjects.json 与 search_items_raw.json。"""
    subjects_file = cfg.subjects_file
    raw_file = cfg.raw_dir / "search_items_raw.json"
    if subjects_file.exists() and subjects_file.stat().st_size > 0 and not force:
        cached = json.loads(subjects_file.read_text(encoding="utf-8"))
        print(f"断点续爬：{subjects_file} 已存在（{len(cached)} 条），跳过采集。")
        print("如需重爬请加 --force。")
        return cached

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    low, high = cfg.douban.page_sleep_sec
    search_text = cfg.douban.search_text
    all_items: list[dict[str, Any]] = []
    subjects: dict[str, dict[str, Any]] = {}
    dropped_person_cards = 0
    dropped_future = 0
    start = 0
    total: int | None = None

    while True:
        if max_pages is not None and start // PAGE_SIZE >= max_pages:
            print(f"已达 --max-pages={max_pages} 上限，停止翻页。")
            break
        print(f"抓取 {SEARCH_URL} start={start} …")
        payload = fetch_page(search_text, start)
        items = payload["items"]
        total = int(payload.get("total") or 0)
        for item in items:
            all_items.append(item)
            year = extract_year(item)
            if year is None:
                dropped_person_cards += 1  # 无 (年份) 后缀：人物卡
                continue
            if year > cfg.douban.max_year:
                dropped_future += 1  # 未上映条目
                continue
            subject_id = str(item["id"])
            if subject_id not in subjects:
                subjects[subject_id] = to_subject(item, year)
        print(
            f"  本页 {len(items)} 条｜累计原始 {len(all_items)}｜"
            f"保留 {len(subjects)}｜人物卡 {dropped_person_cards}"
            f"｜未上映 {dropped_future}｜total={total}"
        )
        start += int(payload.get("count") or PAGE_SIZE)
        if not items or (total and start >= total):
            break
        time.sleep(random.uniform(low, high))

    subject_list = sorted(subjects.values(), key=lambda s: (s["year"], s["id"]))
    subjects_file.write_text(
        json.dumps(subject_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raw_file.write_text(
        json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n清单已写出：{subjects_file}（{len(subject_list)} 条）")
    print(f"原始条目已写出：{raw_file}（{len(all_items)} 条，含被过滤的人物卡等）")
    return subject_list


def run(
    cfg: Config, max_pages: int | None = None, force: bool = False
) -> list[dict[str, Any]]:
    """供 main.py ingest --collect-list 调用。"""
    return collect(cfg, max_pages=max_pages, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="豆瓣搜索列表页采集（urllib，免登录）")
    parser.add_argument("-c", "--config", default="configs/kg_qa.yaml", help="配置文件")
    parser.add_argument(
        "--search-text", default=None, help="覆盖配置里的 douban.search_text"
    )
    parser.add_argument(
        "--max-pages", type=int, default=None, help="最多翻多少页（调试用）"
    )
    parser.add_argument(
        "--force", action="store_true", help="已有 subjects.json 时也重爬"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.search_text:
        cfg.douban.search_text = args.search_text
    collect(cfg, max_pages=args.max_pages, force=args.force)


if __name__ == "__main__":
    main()
