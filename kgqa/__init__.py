"""周星驰电影知识图谱问答系统核心包（v2.2 包化）。

模块地图：
  config.py       配置加载（dataclass + YAML 严格校验）
  llm_client.py   silra API 客户端（理解/生成/抽取）
  embed_client.py SiliconFlow Embedding API 客户端
  collect_list.py 搜索列表采集（urllib，免登录）
  collect_detail.py 详情清洗入库（raw → movies.jsonl）
  build_kg.py     确定性三元组 + 别名表
  graph.py        三元组 → NetworkX 图谱加载
  kg_linker.py    实体链接（别名精确 → 子串 → 向量兜底）
  qa/understanding.py   问题理解（LLM JSON / 规则降级）
  qa/retriever.py       FAISS 双索引（简介分块 + 实体名）
  qa/answering.py       问答编排（路由/直答/RAG/降级）
  extract_llm.py  简介语义三元组抽取（离线）
  evaluate.py     30 题评测
  webapp.py + kg_viz.py Gradio 工作台与图谱可视化
"""
