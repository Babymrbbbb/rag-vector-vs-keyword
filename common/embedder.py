"""Embedder 接口：向量检索的「语义表示」可插拔层。

当前默认实现 `TfidfEmbedder`（纯标准库 TF-IDF 向量空间模型），
作为「关键词检索 (BM25)」的真实对照基线。

未来接入真实语义 embedding 时，只需新增一个实现 `Embedder` 的类
（如基于 sentence-transformers 的 `DenseEmbedder`），并在
`retriever_vector.py` 中替换即可，检索 / 评测 / 报告逻辑完全不动。

接口约定：`embed(text) -> dict[term, float]`
- 稀疏表示（TF-IDF）：键为 token 字符串，值为权重。
- 稠密表示（未来）：键为整型维度下标，值为向量分量。
两者统一用字典承载，余弦相似度对两种表示通用（见 retriever_vector.cosine）。
"""
import math
from abc import ABC, abstractmethod

from .tokenize import tokenize


class Embedder(ABC):
    """所有「文本 -> 权重字典」编码器的抽象基类。"""

    @abstractmethod
    def fit(self, documents):
        """用语料训练（如计算 IDF）。无状态实现可空操作。"""

    @abstractmethod
    def embed(self, text: str) -> dict:
        """把一段文本编码为  term -> weight  的权重字典。"""


class TfidfEmbedder(Embedder):
    """基于词频-逆文档频率（TF-IDF）的稀疏向量表示。"""

    def __init__(self, sublinear_tf: bool = True):
        self.sublinear_tf = sublinear_tf
        self.df = {}      # term -> 出现该词的文档数
        self.n_docs = 0

    def fit(self, documents):
        self.n_docs = len(documents)
        self.df = {}
        for doc in documents:
            for t in set(tokenize(doc)):
                self.df[t] = self.df.get(t, 0) + 1
        return self

    def _idf(self, term: str) -> float:
        # 平滑 IDF，避免未登录词为 0 或负
        df = self.df.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5)) + 1.0

    def embed(self, text: str) -> dict:
        toks = tokenize(text)
        if not toks:
            return {}
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec = {}
        for t, c in tf.items():
            # 次线性 TF 压缩高频词的支配效应
            raw = 1 + math.log(c) if self.sublinear_tf else c
            vec[t] = raw * self._idf(t)
        return vec


class DenseEmbedder(Embedder):
    """真实语义 embedding 占位实现（需安装 sentence-transformers）。

    装好依赖后取消注释即可启用：
        pip install sentence-transformers
    并把 `retriever_vector.py` 中默认 embedder 换成 `DenseEmbedder("BAAI/bge-small-zh")`。
    检索 / 评测 / 报告代码均无需改动 —— 这是本项目的核心可扩展性设计。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "DenseEmbedder 需要 sentence-transformers，当前环境未安装。"
                "请执行 `pip install sentence-transformers` 或使用默认 TfidfEmbedder。"
            ) from exc
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def fit(self, documents):
        return self  # 预训练模型无需训练

    def embed(self, text: str) -> dict:
        self._load()
        vec = self._model.encode(text, normalize_embeddings=True)
        return {i: float(v) for i, v in enumerate(vec)}
