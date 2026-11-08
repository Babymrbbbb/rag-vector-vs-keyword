"""Phase D 评测：BM25（词法） vs chromadb 真·稠密向量（ONNX MiniLM）。

跑 32 条带 gold 标注的查询，算：
- Recall@5（总体 + 按查询类型 exact/colloquial/synonym/longtail）
- 两检索器 top5 结果集 Jaccard
输出控制台 + report/results.json（数字均由真实运行得出，可核）。

运行（用骨架 venv）：
    venv/Scripts/python.exe eval.py
"""
import json
import os
from collections import defaultdict

from retriever_keyword import BM25Retriever, load_corpus
from retriever_vector import ChromaRetriever

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "data", "corpus.jsonl")
QUERIES = os.path.join(HERE, "data", "queries.json")
RESULTS = os.path.join(HERE, "report", "results.json")

TOP_K = 5


def recall_at_k(gold, topk, k):
    if not gold:
        return 0.0
    hit = len(set(gold) & set(topk[:k]))
    return hit / len(gold)


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    docs = load_corpus(CORPUS)
    queries = json.load(open(QUERIES, encoding="utf-8"))

    kw = BM25Retriever().build(docs)
    vec = ChromaRetriever().build(docs)

    per_type = defaultdict(lambda: {"kw": [], "vec": [], "jac": [], "n": 0})
    rows = []
    kw_recalls, vec_recalls, jacs = [], [], []

    for q in queries:
        kw_ids = kw.search(q["q"], TOP_K)
        vec_ids = vec.search(q["q"], TOP_K)
        r_kw = recall_at_k(q["gold"], kw_ids, TOP_K)
        r_vec = recall_at_k(q["gold"], vec_ids, TOP_K)
        jac = jaccard(kw_ids, vec_ids)
        kw_recalls.append(r_kw)
        vec_recalls.append(r_vec)
        jacs.append(jac)
        t = q["type"]
        per_type[t]["kw"].append(r_kw)
        per_type[t]["vec"].append(r_vec)
        per_type[t]["jac"].append(jac)
        per_type[t]["n"] += 1
        rows.append({
            "q": q["q"], "type": t, "gold": q["gold"],
            "kw_topk": kw_ids, "vec_topk": vec_ids,
            "kw_recall": round(r_kw, 3), "vec_recall": round(r_vec, 3),
            "jaccard": round(jac, 3),
        })

    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    overall = {
        "n_queries": len(queries),
        "top_k": TOP_K,
        "kw_recall_mean": _mean(kw_recalls),
        "vec_recall_mean": _mean(vec_recalls),
        "jaccard_mean": _mean(jacs),
        "vector_engine": "chromadb DefaultEmbeddingFunction (ONNX MiniLM-L6-v2, 384d)",
    }
    type_summary = {}
    for t, v in per_type.items():
        type_summary[t] = {
            "n": v["n"],
            "kw_recall_mean": _mean(v["kw"]),
            "vec_recall_mean": _mean(v["vec"]),
            "jaccard_mean": _mean(v["jac"]),
        }

    result = {"overall": overall, "by_type": type_summary, "rows": rows}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---- 控制台输出 ----
    print("=" * 72)
    print(f"评测规模：{overall['n_queries']} 条查询  |  top_k = {TOP_K}")
    print(f"向量引擎：{overall['vector_engine']}")
    print("-" * 72)
    print(f"关键词 BM25        Recall@{TOP_K} 均值 : {overall['kw_recall_mean']:.3f}")
    print(f"向量 稠密(MiniLM)  Recall@{TOP_K} 均值 : {overall['vec_recall_mean']:.3f}")
    print(f"两者结果集 Jaccard 均值           : {overall['jaccard_mean']:.3f}")
    print("-" * 72)
    print("按查询类型细分：")
    print(f"{'type':<11}{'n':>4}{'BM25':>10}{'MiniLM':>10}{'Jaccard':>10}")
    for t in ["exact", "colloquial", "synonym", "longtail"]:
        if t in type_summary:
            s = type_summary[t]
            print(f"{t:<11}{s['n']:>4}{s['kw_recall_mean']:>10.3f}"
                  f"{s['vec_recall_mean']:>10.3f}{s['jaccard_mean']:>10.3f}")
    print("-" * 72)
    print("逐条查询对照（* 表示命中 gold，gold=标准答案 doc id）：")
    for r in rows:
        kw_hit = set(r["gold"]) & set(r["kw_topk"])
        vec_hit = set(r["gold"]) & set(r["vec_topk"])

        def fmt(ids, hit):
            return ",".join(f"{i}{'*' if i in hit else ''}" for i in ids) or "-"

        print(f"[{r['type']:<10}] {r['q']}")
        print(f"   gold   : {r['gold']}")
        print(f"   BM25   : {fmt(r['kw_topk'], kw_hit)}   (R={r['kw_recall']:.2f})")
        print(f"   MiniLM : {fmt(r['vec_topk'], vec_hit)}   (R={r['vec_recall']:.2f})")
    print("=" * 72)
    print(f"结构化结果已写入：{RESULTS}")


if __name__ == "__main__":
    main()
