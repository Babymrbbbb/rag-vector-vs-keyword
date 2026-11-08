# 检索范式对照报告：关键词(BM25) vs 向量(稠密·ONNX MiniLM)

> 本仓库数据为合成演示数据，与任何真实企业经营数据无关；文档只描述方法论与工程实现。

> 本文所有数字均由 `venv/Scripts/python.exe eval.py` 在 `data/` 合成演示语料上实跑得出，
> 原始结构化数据见 [`results.json`](results.json)。**不虚构指标，且结论反直觉——如实呈现。**

## 1. 实验目的

RAG 的检索层是效果天花板。一个常见迷思是「上向量库 + embedding = 检索变强」。
但本实验要验证的更尖锐的问题是：**当 embedding 模型与语料语言/领域不匹配时，
稠密检索是否真的优于词法检索？** 我们用一个 30 篇中文业务知识库 + 32 条带标准答案(gold)的查询，
实跑两种检索器，量化它们在真实中文垂域上的差异。

## 2. 实验设置

| 项 | 值 |
|----|----|
| 语料规模 | 30 篇（合成演示语料，连锁门店运营知识主题） |
| 查询规模 | 32 条，按类型标注：exact 8 / colloquial 7 / synonym 7 / longtail 10 |
| 检索器 A（词法） | **BM25 倒排索引**，中文分词 = unigram + bigram（纯标准库从零实现） |
| 检索器 B（稠密） | **chromadb 默认 ONNX embedding = all-MiniLM-L6-v2（384 维）**，真·语义向量 |
| 评测指标 | Recall@5（gold 文档在 top5 中的召回比例）、结果集 Jaccard（两检索器 top5 重叠度） |

> ⚠️ **关于向量引擎的诚实声明**：本环境 venv 仅含 `chromadb + onnxruntime`，**未装 sentence-transformers/torch**，
> 因此稠密向量用的是 chromadb 自带的英文预训练 MiniLM（首次运行自动下载 79MB 并缓存）。
> 它**确实是真·语义稠密向量**（不是 TF-IDF 之类词法表示），但**不是中文优化模型**——
> 这一点正是本实验最关键的发现来源（见第 5 节）。

## 3. 总体结果（真实运行）

| 检索器 | Recall@5 均值 | 说明 |
|--------|--------------|------|
| 关键词 BM25（中文 unigram+bigram） | **0.953** | 词法，但中文分词到位 |
| 向量 MiniLM（英文预训练·稠密） | **0.594** | 真·语义向量，但语言不匹配 |
| 两者结果集 Jaccard 均值 | **0.298** | top5 高度不重叠（互补而非重复） |

**结论（反直觉但真实）：在本中文垂域小库上，BM25 以 0.953 大幅领先英文预训练 MiniLM 的 0.594。**
稠密向量没有「自动赢」，模型与语言/领域的匹配度，比「是否用向量」更关键。

## 4. 分查询类型结果

| 类型 | 数量 | BM25 Recall@5 | MiniLM Recall@5 | 结果集 Jaccard |
|------|------|--------------|-----------------|----------------|
| exact（字面精确） | 8 | 1.000 | 0.750 | 0.352 |
| colloquial（口语化） | 7 | 0.786 | 0.500 | 0.276 |
| synonym（同义改写） | 7 | 1.000 | **0.429** | 0.294 |
| longtail（长尾组合） | 10 | 1.000 | 0.650 | 0.275 |
| **总体** | **32** | **0.953** | **0.594** | **0.298** |

观察：
- **BM25 在 synonym / longtail 上几乎满分**（1.000 / 1.000），因为中文 unigram+bigram 分词把「面团/运/冷链/加盟」等词面信号抓得很准——垂域用词稳定时词法极其有效。
- **MiniLM 在 synonym 上崩到 0.429**：例如「面团怎么运到店里」(gold=冷链配送)、「蛋糕盒子用什么材料」(gold=包材选择)、「怎么订货第二天能收到」(gold=订货流程)——MiniLM 全部 0 分，返回完全不相关的文档。根源是英文预训练模型把中文切成子词（subword），失去中文语义锚点。
- **两者的 Jaccard 仅 0.298**：它们不是「同一结果两种排序」，而是**高度互补**——这正说明生产环境应做混合检索（hybrid），而非二选一。

## 5. 关键洞察（本实验最大价值）

> **「稠密向量」不是银弹，embedding 模型的语种/领域适配才是。** 一个英文预训练的通用 embedding，
> 放到中文垂域知识库上，Recall@5 从 BM25 的 0.953 跌到 0.594，且在最考验语义的 synonym 类直接腰斩。
> 这与「向量检索一定更强」的流行叙事完全相反，却**实证了 hunkon_rag / kb-cli 的工程判断**：
> 「垂直语料用词稳定，小库场景精确召回 > 语义召回；泛词塌缩用 IDF 降权，长文档刷分用词频饱和。」

### 那语义检索还有价值吗？有，但必须用对模型
要让稠密向量真正赢过 BM25，必须使用**中文优化**的 embedding（如 `BAAI/bge-small-zh-v1.5`）。
本项目的架构已经为此留好**一行切换点**——只需：
```python
# retriever_vector.py 中把 chromadb 默认 embedding 换成：
from sentence_transformers import SentenceTransformer
self.ef = SentenceTransformer("BAAI/bge-small-zh-v1.5")  # 需 pip install sentence-transformers
self.col = self.client.create_collection("swan_kb", embedding_function=self.ef)
```
检索 / 评测 / 报告代码**零改动**。换上 bge 后重跑，预期 synonym/longtail 的 Recall 会显著拉升，
届时再做 **BM25 + bge + RRF 融合** 的混合检索，才是该语料上的最优解。

### 为什么低 Jaccard 是好事
Jaccard=0.298 说明两套检索**抓的是不同文档**：BM25 抓词面命中的，MiniLM 抓它认为"语义近"的。
在真实 RAG 里，这正是 **Reciprocal Rank Fusion（RRF）混合检索** 的价值——把两者结果融合，
既保精确又补语义，互补而非冗余。本实验用数据证明了"为什么要 hybrid"而不是"hybrid 更高级"。

## 6. 架构与可复现

```
rag-vector-vs-keyword-skeleton/
├── data/corpus.jsonl        # 30 篇中文业务语料（示例）
├── data/queries.json        # 32 条查询（type + gold 标注）
├── common/tokenize.py       # 中文分词：unigram + bigram（纯标准库）
├── common/embedder.py       # Embedder 抽象 + TfidfEmbedder + DenseEmbedder 占位
├── retriever_keyword.py     # BM25 倒排（从零实现）
├── retriever_vector.py      # chromadb 真·稠密向量（默认 ONNX MiniLM，可一行换 bge）
├── eval.py                  # 32 查询 × 双检索器，Recall@5 + Jaccard
├── report/comparison.md     # 本报告
├── report/results.json      # 结构化真实结果
└── venv/                    # 含 chromadb + onnxruntime（Python 3.11）
```

运行（用骨架 venv，免装额外依赖）：
```bash
venv/Scripts/python.exe eval.py
```

## 7. 局限与下一步

**局限**
1. 语料仅 30 篇、单领域，Recall@5 在小库上本就偏宽容；
2. 稠密向量用的是**英文预训练** MiniLM（环境无 sentence-transformers），并非中文最优模型——
   因此「BM25 > MiniLM」是"语言不匹配"下的结论，**不代表"词法 > 一切语义"**；
3. gold 标注为示例性人工标注，用于方法对照，非权威评测集。

**下一步（代码接口已就绪，缺运行环境/模型）**
1. `pip install sentence-transformers`，把 `retriever_vector.py` 的 embedding 换成 `BAAI/bge-small-zh-v1.5`；
2. 重跑 `eval.py`，验证中文 embedding 是否反超 BM25（预期 synonym/longtail 大幅提升）；
3. 新增 `hybrid.py`：用 RRF 融合 BM25 + bge，看整体 Recall 是否再上一层；
4. 扩充语料与"难例"查询，做更严格的对照。

## 8. 一句话结论

在可控实验里，我们诚实地看到：**通用英文 embedding 在中文垂域上被精心分词的 BM25 碾压（0.594 vs 0.953），
这反证了"向量一定更强"的迷思，也实证了"小库垂域精确召回优先"的工程判断；真正的语义增益来自
中文优化的 embedding（bge），而本架构已为其预留零改动接入点——下一步用 hybrid(RRF) 收口。**
这为检索层选型提供了可复现、可归因的依据：结论不是来自框架选型偏好，而是来自同一套评测标准下的实测数据。


---

## 复跑说明（2026-11-08）

示例语料与查询中的品牌、地名、门店分布统一换成合成演示值（12 家；A 市 6 / B 市 3 / C 市 2 / D 市 1；E 市无门店）后，
用同一 `eval.py` 重跑 32 条标注查询：BM25 Recall@5 `0.9531`、ONNX MiniLM `0.5938` 均不变；
两路 top5 的 Jaccard 均值由 `0.2628` 变为 `0.2984`，本报告按三位小数记作 `0.298`。
差异来自语料文本改动引起的排序微变，「两路高度互补、应做融合而非二选一」的结论不变。
分类型表格按 `report/results.json` 复核：exact `0.3522`、colloquial `0.2755`、synonym `0.2937`、longtail `0.2746`。
结构化结果见 `report/results.json`。
