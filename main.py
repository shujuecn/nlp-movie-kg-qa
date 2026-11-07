"""命令行入口：ingest / build-kg / index / ask / eval / serve（薄壳，逻辑在 kgqa/ 包）。

子命令与接口契约（docs/接口契约.md §9）一一对应：

  ingest    原始记录 → data/movies.jsonl（kgqa.collect_detail，可选补列表/补采）
  build-kg  结构化字段 → kg/triples.jsonl + kg/aliases.jsonl（kgqa.build_kg.run）
  index     简介分块与实体名/别名 → FAISS 双索引（kgqa.qa.retriever.Retriever.build）
  ask       单条问答：问题理解 → 实体链接 → 直答 / RAG 生成（kgqa.qa.answering.answer）
  eval      30 道金标题评测（kgqa.evaluate.run）
  serve     Gradio 工作台（kgqa.webapp.main）

index / ask / eval / serve 依赖的模块在各子命令分支内延迟导入——
数据管线（ingest / build-kg）不装 faiss / gradio 也能跑。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from kgqa.config import Config, load_config
from kgqa.llm_client import create_client, has_api_key, load_env_file


def read_movies(cfg: Config) -> list[dict[str, Any]]:
    """读取 ingest 产出的 movies.jsonl（index / ask 用）。"""
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


def cmd_ingest(cfg: Config, args: argparse.Namespace) -> None:
    if args.collect_list:
        from kgqa.collect_list import collect

        collect(cfg, max_pages=args.max_pages, force=args.force)
    from kgqa.collect_detail import fetch_missing, ingest

    if args.fetch:
        fetch_missing(
            cfg, storage_state=args.storage_state, headless=not args.no_headless
        )
    ingest(cfg)


def cmd_build_kg(cfg: Config, args: argparse.Namespace) -> None:
    from kgqa.build_kg import run as run_build_kg

    run_build_kg(cfg)


def cmd_index(cfg: Config, args: argparse.Namespace) -> None:
    from kgqa.qa.retriever import Retriever

    movies = read_movies(cfg)
    retriever = Retriever(cfg, cfg.embedding)
    retriever.build(movies)
    print(
        "索引就绪："
        f"{cfg.index_dir / 'faiss_synopsis.faiss'}、"
        f"{cfg.index_dir / 'faiss_entity.faiss'}"
    )


def cmd_ask(cfg: Config, args: argparse.Namespace) -> None:
    from kgqa.graph import load_graph
    from kgqa.kg_linker import load_aliases
    from kgqa.qa.answering import answer
    from kgqa.qa.retriever import Retriever

    llm = None
    if has_api_key(cfg.llm):
        llm = create_client(cfg.llm)
    else:
        print(
            f"[降级模式] 未配置 {cfg.llm.key_env}：事实题模板直答可用，"
            "概述/生成题返回检索证据（配置 key 后可用 RAG 生成）。",
            file=sys.stderr,
        )
    retriever = None
    try:
        retriever = Retriever(cfg, cfg.embedding)
    except Exception as exc:  # 克隆后未建索引/未采集：事实题直答仍可用
        print(
            f"[降级模式] 向量检索不可用（{exc}），简介片段与 RAG 生成停用。",
            file=sys.stderr,
        )
    result = answer(
        args.question,
        graph=load_graph(cfg.kg_dir),
        aliases=load_aliases(cfg.aliases_file),
        retriever=retriever,
        llm=llm,
        cfg=cfg,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_eval(cfg: Config, args: argparse.Namespace) -> None:
    from kgqa.evaluate import run as run_eval

    run_eval(cfg)


def cmd_serve(cfg: Config, args: argparse.Namespace) -> None:
    from kgqa.webapp import main as webapp_main

    argv = ["-c", args.config]
    if args.port is not None:
        argv += ["--port", str(args.port)]
    if args.host is not None:
        argv += ["--host", args.host]
    webapp_main(argv)


def main() -> None:
    parser = argparse.ArgumentParser(description="周星驰电影知识图谱问答系统")
    parser.add_argument("-c", "--config", default="configs/kg_qa.yaml", help="配置文件")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="原始记录入库 → data/movies.jsonl")
    p_ingest.add_argument(
        "--collect-list", action="store_true", help="先重跑搜索列表采集"
    )
    p_ingest.add_argument(
        "--max-pages", type=int, default=None, help="列表最多翻多少页（调试）"
    )
    p_ingest.add_argument(
        "--force", action="store_true", help="已有 subjects.json 时也重爬"
    )
    p_ingest.add_argument(
        "--raw-dir",
        default=None,
        help="覆盖配置的 douban.raw_dir（debug：读指定原始目录）",
    )
    p_ingest.add_argument(
        "--fetch", action="store_true", help="缺失单片用 playwright 登录态补采"
    )
    p_ingest.add_argument(
        "--storage-state", default=None, help="playwright storage_state 文件"
    )
    p_ingest.add_argument(
        "--no-headless", action="store_true", help="补采时显示浏览器窗口"
    )

    sub.add_parser("build-kg", help="结构化字段 → 确定性三元组 + 别名表")
    sub.add_parser("index", help="构建 FAISS 双索引（简介分块 + 实体名）")
    p_ask = sub.add_parser("ask", help="单条问答，输出 JSON 结果")
    p_ask.add_argument("-q", "--question", required=True, help="问题文本")
    sub.add_parser("eval", help="跑 30 道金标题评测")
    p_serve = sub.add_parser("serve", help="启动 Gradio 工作台")
    p_serve.add_argument(
        "--port", type=int, default=None, help="覆盖配置的 gradio.port"
    )
    p_serve.add_argument("--host", default=None, help="覆盖配置的 gradio.host")

    args = parser.parse_args()
    load_env_file()
    cfg = load_config(args.config)
    if args.command == "ingest" and args.raw_dir:
        cfg.douban.raw_dir = args.raw_dir
    {
        "ingest": cmd_ingest,
        "build-kg": cmd_build_kg,
        "index": cmd_index,
        "ask": cmd_ask,
        "eval": cmd_eval,
        "serve": cmd_serve,
    }[args.command](cfg, args)


if __name__ == "__main__":
    main()
