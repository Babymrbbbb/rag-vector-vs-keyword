"""可选：接中文 bge 的入库脚本（Step 1）。

说明：默认评测管线用的是 `retriever_vector.py`（chromadb 自带 ONNX MiniLM，免装 sentence-transformers）。
若要换成中文优化的 BAAI/bge-small-zh-v1.5，请先 `pip install chromadb sentence-transformers`，
并把本脚本的 `SentenceTransformer(EMBED_MODEL)` 与 retriever_vector.py 的 embedding_function 同步替换。
本文件仅作为「切到中文 embedding」的参考入口，默认评测无需运行它。

依赖：pip install chromadb sentence-transformers
用法：python ingest.py --data data/raw
"""
import os
import glob
import argparse

from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # 本地中文 embedding，无需联网调用


def chunk_text(text, max_chars=800, overlap=80):
    """简单按长度切块，保留少量重叠，避免跨句切断语义。"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + max_chars])
        start += max_chars - overlap
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    args = ap.parse_args()

    client = Client(Settings(anonymized_telemetry=False))
    try:
        client.delete_collection("kb")
    except Exception:
        pass
    col = client.create_collection("kb")

    model = SentenceTransformer(EMBED_MODEL)

    docs, ids, metas = [], [], []
    for i, fp in enumerate(glob.glob(os.path.join(args.data, "**", "*.md"), recursive=True)):
        text = open(fp, encoding="utf-8").read()
        for ci, chunk in enumerate(chunk_text(text)):
            docs.append(chunk)
            ids.append(f"{i}-{ci}")
            metas.append({"source": os.path.basename(fp)})

    col.add(
        documents=docs,
        ids=ids,
        metadatas=metas,
        embeddings=model.encode(docs).tolist(),
    )
    print(f"入库完成：{len(docs)} 个块，来源文件 {len(set(m['source'] for m in metas))} 个")


if __name__ == "__main__":
    main()
