"""Step 3: 倒排索引检索（BM25，纯标准库，从零实现）。

与 retriever_vector.py（chromadb 真·稠密向量）形成真实对照。
接口：BM25Retriever().build(docs).search(q, k) -> List[int]（doc id 列表）

用法：python retriever_keyword.py "会员年费"
"""
import json
import math
from collections import defaultdict

from common.tokenize import tokenize


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = {}                  # doc_id -> {id,title,text}
        self.doc_len = {}               # doc_id -> token 数
        self.avgdl = 0.0
        self.df = defaultdict(int)      # term -> 含该词的文档数
        self.inverted = defaultdict(list)  # term -> [(doc_id, tf)]

    def build(self, documents: list[dict]):
        self.docs = {d["id"]: d for d in documents}
        total = 0
        for d in documents:
            toks = tokenize(d.get("title", "") + " " + d.get("text", ""))
            self.doc_len[d["id"]] = len(toks)
            total += len(toks)
            tf = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, c in tf.items():
                self.df[t] += 1
                self.inverted[t].append((d["id"], c))
        n = len(documents) or 1
        self.avgdl = total / n
        return self

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (len(self.docs) - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[int]:
        q_toks = set(tokenize(query))
        if not q_toks:
            return []
        scores = defaultdict(float)
        for t in q_toks:
            idf = self._idf(t)
            if idf <= 0:
                continue
            for doc_id, tf in self.inverted.get(t, []):
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [doc_id for doc_id, _ in ranked[:top_k]]


def load_corpus(path: str) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def query(q: str, top_k: int = 5):
    """CLI 兼容：返回 (documents, metadatas)，便于独立调试。"""
    docs = load_corpus("data/corpus.jsonl")
    r = BM25Retriever().build(docs)
    ids = r.search(q, top_k)
    texts = [r.docs[i]["text"] for i in ids]
    metas = [{"doc_id": i, "source": r.docs[i]["title"]} for i in ids]
    return texts, metas


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("q", help="查询问题")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    texts, metas = query(args.q, args.k)
    for d, m in zip(texts, metas):
        print(f"[{m['source']} | id={m['doc_id']}]\n{d[:200]}\n---")
