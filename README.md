# 周星驰电影知识图谱问答系统

采集豆瓣「周星驰」电影专题（入库 73 部），抽取三元组构建 NetworkX 知识图谱，配 FAISS 双索引
（简介分块 RAG + 实体名/别名向量兜底）做混合问答：事实题**模板直答**（本地、零成本、毫秒级），
概述/生成题把子图与检索片段交给 silra API（`qwen3.8-flash`）**RAG 生成**并要求引用证据；
Embedding 用 SiliconFlow 的 `Qwen/Qwen3-Embedding-8B`，全 API、零深度学习依赖、纯 CPU 可跑；
承接 261101：向量走向「检索」，图谱负责「事实」，API 大模型负责「理解与表达」。

```
豆瓣「周星驰」专题（122 条 → 入库 73 部；列表页 requests 免登录，详情页登录态浏览器限速采集）
  └─▶ data/movies.jsonl（全字段 + 简介）──▶ kgqa/build_kg.py（确定性三元组 + 实体别名表）
        └─▶ kgqa/extract_llm.py（语义三元组：角色-饰演者、题材；备课期离线抽取，产物随仓库分发）
  本地（CPU 即可）：NetworkX 图谱谓词查询 ｜ FAISS 简介分块 + 实体名/别名双索引 ｜ Gradio 工作台
  silra API（qwen3.8-flash）：① 离线语义三元组抽取 ② 在线问题理解（均 JSON mode）③ 在线 RAG 生成（要求引用）
```

## 0. 项目结构

```
├── main.py                # 唯一 CLI 入口（薄壳）：ingest/build-kg/index/ask/eval/serve
├── kgqa/                  # 核心包
│   ├── config.py          #   dataclass + YAML 严格校验（未知字段报错）
│   ├── llm_client.py      #   silra API 客户端（理解/生成/抽取）
│   ├── embed_client.py    #   SiliconFlow Embedding API 客户端
│   ├── collect_list.py    #   搜索列表采集（urllib，免登录，断点续爬）
│   ├── collect_detail.py  #   详情清洗入库（playwright 补采、剧集/综艺过滤、片长清洗）
│   ├── build_kg.py        #   结构化字段 → 确定性三元组 + 别名表
│   ├── graph.py           #   三元组 → NetworkX 图谱加载
│   ├── kg_linker.py       #   实体链接：别名精确 → 子串 → 向量兜底
│   ├── qa/understanding.py    # 问题理解（LLM JSON mode / 规则降级）
│   ├── qa/retriever.py        # FAISS 双索引 + 检索排序演示（python -m 直跑）
│   ├── qa/answering.py        # 问答编排：路由/直答/RAG/降级/兜底
│   ├── extract_llm.py     #   简介语义三元组抽取（--all / --subject）
│   ├── evaluate.py        #   30 题评测（三指标）
│   └── webapp.py kg_viz.py    # Gradio 工作台 + 图谱可视化
├── configs/               # kg_qa.yaml（正式）+ debug*.yaml（调试沙箱）
├── data/                  # 全部数据（随仓库分发，见 §3）
├── docs/                  # 研究计划、接口契约
└── .vscode/launch.json    # 5 个调试配置（数据采集/清洗/抽取/RAG 检索/答案生成）
```

## 1. 环境准备

conda + Python 3.11（环境名随意）。**本地无需 GPU、无需 torch**——LLM 与 Embedding 全走 API：

```bash
conda create -n kgqa python=3.11 -y && conda activate kgqa
pip install -r requirements.txt
```

两个 key 写进根目录 `.env`（复制 `.env.example` 后填入，`.env` 已 gitignore）：`SILRA_API_KEY`（LLM，silra.cn 兑换）、
`SILICONFLOW_API_KEY`（Embedding，cloud.siliconflow.cn 获取）。VSCode 可用 `.vscode/launch.json` 的 5 个调试配置单步观察各阶段，写入 `data/debug/` 沙箱。

## 2. 五步跑通

```bash
python main.py ingest      # ① 原始详情入库
python main.py build-kg    # ② 结构化字段 → 确定性三元组 + 实体别名表
python main.py index       # ③ 构建 FAISS 双索引（简介分块 + 实体名/别名）
python main.py ask -q "功夫是谁导演的"   # ④ 命令行单问（另有 eval 子命令）
python main.py serve       # ⑤ 网页工作台，http://127.0.0.1:7860
```

各步实测输出（2026-11-08 全链路验收）：
1. `ingest`：合并 `data/raw/*.json` → `data/movies.jsonl`，入库 73 部、过滤 49 条（明细 `data/ingest_report.json`）；
2. `build-kg`：三元组 1389 条、别名 486 行 → `data/kg/`；语义抽取合并后 963 节点 / 1692 边（含 `data/kg/triples_llm.jsonl` 303 条 饰演/题材）。
3. `index`：简介分块 83 块 / 实体 966 个 → `data/index/faiss_*.faiss`（约 18k prompt tokens）。
4. `ask`：输出 JSON（mode / answer / evidence）。事实题模板直答（mode=direct），生成题 RAG 带引用（mode=rag）。
5. `serve`：工作台打开即自动演示一条例子，并在输入框预填下一问。

> **克隆即用**：`data/` 全部随仓库分发，克隆后 ②③④ 无需采集即可跑；首次 `index` / `ask` 会自动重建 FAISS 索引（约 18k tokens，几分钱），事实题直答连 key 都不需要。

## 3. 数据放哪与再采集

```
data/                        # 全部数据，随仓库分发
├── raw/                     # 采集原始页（122 部单片 JSON + subjects.json 清单）
├── movies.jsonl             # 清洗入库（全字段 + 剧情简介）
├── ingest_report.json       # ingest 过滤/缺失明细
├── kg/                      # 图谱成品
│   ├── triples.jsonl        #   确定性三元组（source=douban-info）
│   ├── triples_llm.jsonl    #   离线语义三元组：角色-饰演者、题材（source=llm-offline）
│   └── aliases.jsonl        #   实体别名表（电影又名、周星驰英文名/昵称）
├── eval/                    # 评测集 qa_eval.jsonl（30 题带金标）
├── index/                   # FAISS 索引（gitignore，main.py index 自动重建）
├── eval_report.json         # 评测报告（gitignore，main.py eval 生成）
└── debug/                   # VSCode 调试沙箱（gitignore）
```
**详情页登录墙**（2026-11-07 实测）：未登录 requests 直连会跳 `sec.douban.com` 安全页
（`error code: 01004 Please login`），搜索列表页仍免登录。学生复现有两条路：
1. **Agent 驱动浏览器**：在自己的 Agent Harness（ZCode / Claude Code 等）里登录豆瓣一次，逐部采集详情存 `data/raw/<subject_id>.json`（见 `课后作业.md` 任务一）；
2. **playwright storage_state**：手动登录一次并保存登录态文件后复用：
   ```bash
   python -m kgqa.collect_list                     # 搜索列表：requests 免登录
   python -m kgqa.collect_detail --fetch --storage-state s.json  # 缺失单片用登录态补采
   python -m kgqa.collect_detail                   # 只合并已落盘的原始记录
   ```

## 4. 四种回答模式

同一问题四种证据组装方式，工作台右下单选切换（`answer(force_route=...)`，契约 §6）：

| 模式 | 行为 | API 成本 |
| --- | --- | --- |
| 自动 | 智能路由：问题理解 → 事实题直答 / 概述题 RAG | 按需 |
| 仅直答 | 本地模板直答（规则理解），跳过生成 | 0 |
| 仅 RAG | 强制走生成路线（与直答对照） | 2 次/问 |
| 仅检索 | 只返回证据不生成（三元组 + 检索片段） | 0 |

## 5. 网页工作台

- **自动示例**：`python main.py serve` 打开 http://127.0.0.1:7860 即演示一条例子，输入框预填下一问；
- **状态签**：标题行右侧实时显示 运行中/仅直答、两个模型名、语料规模；
- **证据面板**：问题理解 JSON、命中三元组（类型着色胶囊 + 来源角标）、检索片段（score 徽章 + 强度条，点击展开/收起）、本次 API 用量；
- **引用角标**：答案里的 `[n]` 可点击，高亮右侧对应证据卡；
- **图谱浏览抽屉**（占 80% 宽）：左三元组数据表（1692 条，搜索即过滤、行点击聚焦），
  右 d3 力导向邻域图（类型着色、可拖拽、点击高亮邻居）；点实体/关系胶囊弹出聚焦邻域网络图。

## 6. 配置

全部配置集中在 `configs/kg_qa.yaml`（约 60 行，逐项带注释），未知字段直接报错；命令行可用 `--host/--port` 覆盖工作台地址（如 `python main.py serve --port 7861`）。

## 7. 评测

`python main.py eval` 逐题调 `kgqa.qa.answering.answer`，报告写 `data/eval_report.json`；评测集 `data/eval/qa_eval.jsonl` 共 30 道带金标的题，按检验能力分五类：

| 类型 | 数量 | 检验点 |
| --- | --- | --- |
| fact | 12 | 单关系查询（导演 / 评分 / 年份 / 主演），模板直答 |
| multi_hop | 6 | 反向边反查与交集（周星驰导演作品、吴孟达合作片单等） |
| role | 6 | 「角色-饰演者」语义三元组（紫霞仙子→朱茵、周小狄→徐娇等） |
| overview | 4 | RAG 生成（剧情概述），检查引用与检索片段相关 |
| out_of_scope | 2 | 超范围礼貌兜底（天气、写诗） |

三项指标（延续「每课可度量」，实测 2026-11-08）：
1. **实体链接命中率** **1.0000**（28/28，两道超范围题无实体提及不计入）；
2. **答案关键词命中率** **0.9333**（28/30；未命中的两题是金标词为空的超范围题，有金标词的题 28/28 全中）；
3. **生成题引用有效率** **1.0000**（4/4，mode=rag 的题答案含 `[n]` 且 n 对应证据存在）。

## 8. 无 key 降级模式

延续「克隆即可跑通」的底线：两个 key 独立降级，工作台状态签显示「◐ 降级运行」——

| 场景 | 行为 |
| --- | --- |
| 缺 `SILRA_API_KEY`（LLM） | 事实题模板直答照常；概述/生成题 `mode=degraded`，直接返回检索证据；问题理解走规则降级（书名号/引号与别名表找实体） |
| 缺 `SILICONFLOW_API_KEY`（Embedding） | 向量兜底与简介检索不可用——事实题直答不受影响，生成题直接 `mode=degraded` |
| 两个都缺 | 事实题直答 + 证据面板展示检索结果（三元组来自 `data/kg/`，随仓库分发） |

## 9. 常见问题

1. **豆瓣详情页 01004 登录墙**：未登录 requests 直连会 302 到 `sec.douban.com` 安全页，详情采集必须走登录态浏览器（两条复现路径见 §3），列表页不受影响。
2. **FAISS 索引加载失败**：找不到 `data/index/faiss_*.faiss` 时自动重建（需 `SILICONFLOW_API_KEY`）；换 embedding 模型/维度同理，也可手动 `python main.py index`。
3. **gradio 7860 端口占用**：`python main.py serve --port 7861`。
4. **API 超时 / 限流**：默认 60 秒超时、重试 2 次（线性退避），仍失败返回中文报错；离线抽取约 60k tokens、索引约 18k、单次 RAG 2-4k，成本几分钱级。
5. **RAG 证据里混入了别的电影**（经典教学案例）：短查询的向量被题材语义主导——实测全库 83 个简介块相似度挤在 0.445~0.514，《功夫》本体块排第 6，纯语义 top-3 是《1:99电影行动》《美猴王》《武状元苏乞儿》。
   保底机制：实体链接命中电影后，其自身简介块强制提升到证据首位（`kgqa/qa/answering._promote_linked_movie_passages`）；课堂可现场对比「纯语义排序」与「实体过滤检索」两种证据组装方式。

## 10. 开发与来源

```bash
ruff format . && ruff check .
mypy main.py kgqa/
```

- 接口契约见 `docs/接口契约.md`，项目演进见 `docs/研究计划.md`；`config.py` 严格校验、`webapp.py` 的工程习惯沿用 261101。
- 模型：Embedding 用 SiliconFlow [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B)（MRL 1024 维）；
  理解与生成用 silra API 的 `qwen3.8-flash`（OpenAI 兼容网关，[帮助文档](https://silra.apifox.cn/)）。
- **数据来源声明**：电影条目与详情采集自**豆瓣**（movie.douban.com），限速礼貌采集，**仅教学使用，请勿二次转载**；
  原始页快照与剧情简介随仓库一并分发于 `data/`，商用前请自行核对豆瓣的使用条款。
