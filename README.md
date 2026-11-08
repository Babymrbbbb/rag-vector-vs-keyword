# rag-vector-vs-keyword · 关键词(BM25) vs 向量(稠密) 检索对照

> 本仓库数据为合成演示数据，与任何真实企业经营数据无关；文档只描述方法论与工程实现。

> 阶段 D · M1 展示型 RAG 对照实验。用同一套中文业务语料 + 32 条带标准答案的查询，
> 实跑 **BM25 词法检索** vs **chromadb 真·稠密向量**，量化两者在真实中文垂域上的差距。
> 结论反直觉但真实，直接实证"小库垂域精确召回优先"的工程判断。

## 一句话结论（真实运行）

在 30 篇中文垂域语料、32 条查询上：**BM25 Recall@5 = 0.953，英文预训练 MiniLM 稠密向量 Recall@5 = 0.594**。
通用英文 embedding 在中文垂域上被精心分词的 BM25 碾压；真正的语义增益来自**中文优化的 embedding（bge）**——
本项目已为其预留零改动接入点。详见 [`report/comparison.md`](report/comparison.md)。

## 目录结构

```
rag-vector-vs-keyword-skeleton/
├── data/corpus.jsonl        # 30 篇中文业务语料（示例）
├── data/queries.json        # 32 条查询（type + gold 标准答案标注）
├── common/tokenize.py       # 中文分词：unigram + bigram（纯标准库）
├── common/embedder.py       # Embedder 抽象 + TfidfEmbedder + DenseEmbedder 占位
├── retriever_keyword.py     # BM25 倒排索引（从零实现，零依赖）
├── retriever_vector.py      # chromadb 真·稠密向量（默认 ONNX MiniLM，可一行换 bge）
├── eval.py                  # 32 查询 × 双检索器，Recall@5 + Jaccard
├── ingest.py                # 可选：接中文 bge 的入库脚本（需 sentence-transformers）
├── report/comparison.md     # 真人分析报告（真实数字，含失败案例）
├── report/results.json      # 结构化真实结果
└── venv/                    # 已含 chromadb + onnxruntime（Python 3.11，免装即跑）
```

## 运行（用自带 venv，免装额外依赖）

```bash
cd rag-vector-vs-keyword-skeleton
venv/Scripts/python.exe eval.py        # 跑对照，输出总体/分类型 Recall@5 + Jaccard
venv/Scripts/python.exe retriever_keyword.py "会员年费"   # 试一次 BM25
venv/Scripts/python.exe retriever_vector.py "会员年费"    # 试一次稠密向量
```

> 首次运行会自动下载 chromadb 默认 ONNX MiniLM 模型（79MB）并缓存，之后离线可用。

## 换成中文 embedding（让稠密真正赢）

本环境 venv 未装 `sentence-transformers`，故稠密向量用的是英文预训练 MiniLM。
要换成中文优化的 `BAAI/bge-small-zh-v1.5`，只需在 `retriever_vector.py` 改一处：

```python
from sentence_transformers import SentenceTransformer
self.ef = SentenceTransformer("BAAI/bge-small-zh-v1.5")          # 新增
self.col = self.client.create_collection("swan_kb", embedding_function=self.ef)  # 传入
```

检索 / 评测 / 报告代码**零改动**，重跑 `eval.py` 即可看到 synonym/longtail 召回是否反超 BM25，
届时再加 RRF 混合检索收口。

## 结论与工程判断

- **用数据推翻了一个默认假设**：「是否用向量」不是检索质量的决定因素，模型与语言/领域的匹配度才是。
- **给出可复现的适用条件**：垂直语料用词稳定、规模有限时，精确字面召回优于语义召回（0.953 vs 0.594）。
- **两路结果高度互补**（结果集重合度仅 0.298），说明生产里应做两路融合而非二选一；该判断已在
  [langchain-kb-qa](https://github.com/Babymrbbbb/langchain-kb-qa) 中落地验证。
- **表示层做成可替换**：换中文优化模型只需改一处，检索/评测/报告零改动，便于持续对照。

## 数据说明

`data/corpus.jsonl` 与 `data/queries.json` 为**合成演示语料**（围绕连锁门店运营知识主题构造），
仅用于检索方法对照，不含任何真实经营数据、个人隐私、密钥或内部系统路径。

> 本仓库数据为合成演示数据，与任何真实企业经营数据无关；文档只描述方法论与工程实现。
