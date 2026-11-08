"""Step 2: 向量检索（chromadb + 默认 ONNX embedding，真·稠密向量）。

与 retriever_keyword.py（BM25 词法）形成**真实**对照：本检索器用的是 chromadb
自带的 DefaultEmbeddingFunction（ONNX MiniLM-L6-v2，384 维），是真正的语义向量，
不是 TF-IDF 之类的词法表示——因此本实验才是原规划想要的「词法 vs 语义」对照。

环境说明：本骨架 venv 仅含 chromadb + onnxruntime，**未装 sentence-transformers/torch**，
故用 chromadb 内置 ONNX MiniLM（首次运行自动下载 79MB 并缓存，之后离线可用）。
若要换成中文 bge 以获得更准的中文语义，只需：
    pip install sentence-transformers
然后把下面 ChromaRetriever 的 embedding_function 换成
    SentenceTransformer("BAAI/bge-small-zh-v1.5")
（接口完全一致，检索/评测代码无需改动）。

接口：ChromaRetriever().build(docs).search(q, k) -> List[int]（doc id 列表）

用法：python retriever_vector.py "会员年费"
"""
import json

from chromadb import Client


class ChromaRetriever:
    def __init__(self, collection_name: str = "swan_kb"):
        self.collection_name = collection_name
        self.client = Client()           # 内存客户端，评测内重建一次即可
        self.doc_index = {}              # chroma_id -> doc_id
        self.col = None
        self._docs_by_id = {}

    def build(self, documents: list[dict]):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        # 不显式传 embedding_function -> chromadb 用默认 ONNX MiniLM（与 add/query 一致）
        self.col = self.client.create_collection(self.collection_name)
        chroma_ids, docs, metas = [], [], []
        for d in documents:
            cid = f"doc_{d['id']}"
            self.doc_index[cid] = d["id"]
            self._docs_by_id[d["id"]] = d
            chroma_ids.append(cid)
            docs.append(d.get("title", "") + "\n" + d.get("text", ""))
            metas.append({"doc_id": d["id"], "source": d.get("title", "")})
        self.col.add(documents=docs, ids=chroma_ids, metadatas=metas)
        return self

    def search(self, query: str, top_k: int = 5) -> list[int]:
        if self.col is None:
            raise RuntimeError("请先调用 build()")
        res = self.col.query(query_texts=[query], n_results=top_k)
        ids = res["ids"][0]
        return [self.doc_index[i] for i in ids]


def load_corpus(path: str) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def query(q: str, top_k: int = 5):
    docs = load_corpus("data/corpus.jsonl")
    r = ChromaRetriever().build(docs)
    ids = r.search(q, top_k)
    texts = [r._docs_by_id[i]["text"] for i in ids]
    metas = [{"doc_id": i, "source": r._docs_by_id[i]["title"]} for i in ids]
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
