import argparse
import math
import os
import pickle
import torch
import re
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from concurrent.futures import ThreadPoolExecutor

from isection_pipeline import get_embedding_model
from tg_connection import get_tg_connection

# Initialize worker thread pool for parallel retrieval
_RETRIEVAL_POOL = ThreadPoolExecutor(max_workers=2)  
# Configuration
RETRIEVAL_THRESHOLD = 0.35
SEARCH_QUERY = "chunk_search"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

RRF_K = 60
DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 30

AGGREGATION_TOP_K = 30
AGGREGATION_FETCH_K = 70

BM25_INDEX_FILENAME = "bm25_index.pkl"
BM25_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), BM25_INDEX_FILENAME)

_CONN = None
_RERANKER = None
_BM25 = None
_BM25_META = None
_WARNED_NO_DISTANCE = False
_WARNED_NO_BM25 = False


def _conn():
    global _CONN
    if _CONN is None:
        _CONN = get_tg_connection()
    return _CONN


def get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker optimized for CPU execution."""
    global _RERANKER
    if _RERANKER is None:
        torch.set_num_threads(4)
        print(f"  [Reranker] Loading '{RERANKER_MODEL_NAME}' on CPU (Threads=4, MaxLength=256)...")
        _RERANKER = CrossEncoder(
            RERANKER_MODEL_NAME, 
            max_length=256,
            device="cpu"
        )
    return _RERANKER


def _load_bm25_index() -> Tuple[Optional[BM25Okapi], Optional[List[Dict[str, Any]]]]:
    global _BM25, _BM25_META, _WARNED_NO_BM25

    if _BM25 is not None and _BM25_META is not None:
        return _BM25, _BM25_META

    if not os.path.exists(BM25_INDEX_PATH):
        if not _WARNED_NO_BM25:
            print(
                f"  [BM25] Index not found at '{BM25_INDEX_PATH}'. "
                "Run isection_pipeline.py once to build the corpus-wide BM25 index."
            )
            _WARNED_NO_BM25 = True
        return None, None

    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            payload = pickle.load(f)

        tokenized_corpus = payload["tokenized_corpus"]
        metadata = payload["metadata"]

        _BM25 = BM25Okapi(tokenized_corpus)
        _BM25_META = metadata
        print(f"  [BM25] Loaded {len(metadata)} indexed chunks.")
        return _BM25, _BM25_META

    except Exception as exc:
        if not _WARNED_NO_BM25:
            print(f"  [BM25] Failed to load index: {exc}")
            _WARNED_NO_BM25 = True
        return None, None


def extract_query_years(text: str) -> List[str]:
    return re.findall(r"\b(?:19|20)\d{2}\b", text)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def is_aggregation_query(query: str) -> bool:
    q = query.lower()
    triggers = [
        "how many",
        "list all",
        "total events",
        "highest number",
        "all events",
        "most competitors",
        "which had the",
        "how many events",
        "count",
        "all the",
        "every event",
    ]
    return any(trigger in q for trigger in triggers)


def get_dynamic_k(query: str, default_k: int = DEFAULT_TOP_K) -> int:
    return AGGREGATION_TOP_K if is_aggregation_query(query) else default_k


def get_dynamic_fetch_k(query: str, default_k: int = DEFAULT_FETCH_K) -> int:
    return AGGREGATION_FETCH_K if is_aggregation_query(query) else default_k


def _parse_distances(obj: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    elif isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if isinstance(key, dict):
                key = key.get("v_id")
            if key is not None and "value" in item:
                try:
                    out[str(key)] = float(item["value"])
                except (TypeError, ValueError):
                    pass

    return out


def _parse_search_response(res: Any) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    vertices: List[Dict[str, Any]] = []
    distances: Dict[str, float] = {}

    for block in res or []:
        if not isinstance(block, dict):
            continue

        for key, value in block.items():
            if key == "@@distances":
                distances = _parse_distances(value)
            elif (
                isinstance(value, list)
                and value
                and isinstance(value[0], dict)
                and "v_id" in value[0]
            ):
                vertices = value

    return vertices, distances


def vector_search(query_embedding: list, k: int) -> List[Tuple[str, str, str, Optional[float]]]:
    try:
        response = _conn().runInstalledQuery(
            SEARCH_QUERY,
            {
                "query_vector": [float(x) for x in query_embedding],
                "k": int(k),
            },
        )

        vertices, distances = _parse_search_response(response)
        results = []

        for vertex in vertices:
            attrs = vertex.get("attributes", {})
            chunk_id = vertex.get("v_id")
            if not chunk_id:
                continue

            results.append(
                (
                    str(chunk_id),
                    attrs.get("content", "") or "",
                    attrs.get("source", "unknown"),
                    distances.get(str(chunk_id)),
                )
            )

        return results

    except Exception as exc:
        print(f"  [Vector Error] Query execution failed: {exc}")
        return []


def bm25_search(query: str, k: int) -> List[Tuple[str, float]]:
    bm25, metadata = _load_bm25_index()
    if bm25 is None or metadata is None:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: float(scores[idx]),
        reverse=True,
    )[: int(k)]

    results = []
    for idx in ranked_indices:
        score = float(scores[idx])
        chunk_id = metadata[idx]["chunk_id"]
        results.append((chunk_id, score))

    return results


def _rrf(rank_zero_based: int) -> float:
    return 1.0 / (RRF_K + rank_zero_based + 1)


def build_hybrid_candidates(
    dense_hits: List[Tuple[str, str, str, Optional[float]]],
    bm25_hits: List[Tuple[str, float]],
) -> List[Dict[str, Any]]:
    if not dense_hits and not bm25_hits:
        return []

    dense_rank = {
        chunk_id: rank
        for rank, (chunk_id, _, _, _) in enumerate(dense_hits)
    }
    bm25_rank = {
        chunk_id: rank
        for rank, (chunk_id, _) in enumerate(bm25_hits)
    }

    by_id: Dict[str, Dict[str, Any]] = {}

    for chunk_id, content, source, distance in dense_hits:
        by_id[chunk_id] = {
            "content": content,
            "source": source,
            "distance": distance,
        }

    bm25_lookup = {
        chunk_id: score
        for chunk_id, score in bm25_hits
    }

    _, metadata = _load_bm25_index()
    metadata_by_id = (
        {item["chunk_id"]: item for item in metadata}
        if metadata is not None
        else {}
    )
    for chunk_id, _ in bm25_hits:
        if chunk_id in by_id:
            continue
        item = metadata_by_id.get(chunk_id)
        if item is None:
            continue
        by_id[chunk_id] = {
            "content": item.get("content", ""),
            "source": item.get("source", "unknown"),
            "distance": None,
        }

    candidates: List[Dict[str, Any]] = []

    for chunk_id, record in by_id.items():
        fused_score = 0.0

        if chunk_id in dense_rank:
            fused_score += _rrf(dense_rank[chunk_id])

        if chunk_id in bm25_rank:
            fused_score += _rrf(bm25_rank[chunk_id])

        if fused_score <= 0:
            continue

        distance = record.get("distance")
        semantic_score = (
            0.0
            if distance is None
            else max(0.0, 1.0 - float(distance))
        )

        candidates.append(
            {
                "content": record.get("content", ""),
                "metadata": {
                    "source": record.get("source", "unknown"),
                    "chunk_id": chunk_id,
                },
                "cosine_similarity": semantic_score,
                "bm25_score": float(bm25_lookup.get(chunk_id, 0.0)),
                "dense_rank": dense_rank.get(chunk_id),
                "bm25_rank": bm25_rank.get(chunk_id),
                "rrf_score": float(fused_score),
                "hybrid_score": float(fused_score),
                "final_score": float(fused_score),
            }
        )

    if not candidates:
        return []

    max_rrf = max(item["rrf_score"] for item in candidates)
    for item in candidates:
        item["hybrid_score"] = (
            item["rrf_score"] / max_rrf if max_rrf > 0 else 0.0
        )
        item["final_score"] = item["hybrid_score"]

    candidates.sort(key=lambda item: item["rrf_score"], reverse=True)
    return candidates


def apply_year_penalty(query: str, text: str, current_score: float) -> float:
    q_lower = query.lower()
    q_years = extract_query_years(query)
    if not q_years:
        return current_score

    target_year = int(q_years[0])
    doc_years = [int(y) for y in extract_query_years(text)]

    if not doc_years:
        return current_score

    if "before" in q_lower:
        if not any(y < target_year for y in doc_years):
            return 0.0
        return current_score

    if "after" in q_lower:
        if not any(y > target_year for y in doc_years):
            return 0.0
        return current_score

    if target_year not in doc_years:
        return current_score * 0.3

    return current_score


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [[query, item["content"]] for item in candidates]
    scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)

    for item, raw_score in zip(candidates, scores):
        raw_score = float(raw_score)
        rerank_score = 1.0 / (1.0 + math.exp(-raw_score))
        item["rerank_score"] = rerank_score

        combined = (0.75 * rerank_score) + (0.25 * item["hybrid_score"])
        item["final_score"] = apply_year_penalty(
            query,
            item["content"],
            combined,
        )

    candidates.sort(key=lambda item: item["final_score"], reverse=True)
    return candidates


def hybrid_search(
    query: str,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    enable_rerank: bool = True,
    min_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    global _WARNED_NO_DISTANCE

    embedder = get_embedding_model()
    query_text = query

    if (
        hasattr(embedder, "model_name")
        and "nomic" in getattr(embedder, "model_name", "").lower()
        and not query_text.startswith("search_query:")
    ):
        query_text = f"search_query: {query}"

    effective_top_k = (
        get_dynamic_k(query)
        if top_k is None
        else int(top_k)
    )
    effective_fetch_k = (
        get_dynamic_fetch_k(query)
        if fetch_k is None
        else int(fetch_k)
    )

    query_embedding = embedder.embed_query(query_text)

    # --- OPTIMIZATION 2: Parallel execution of Vector and BM25 searches ---
    dense_future = _RETRIEVAL_POOL.submit(vector_search, query_embedding, effective_fetch_k)
    bm25_future = _RETRIEVAL_POOL.submit(bm25_search, query, effective_fetch_k)

    dense_hits = dense_future.result()
    bm25_hits = bm25_future.result()

    if not dense_hits and not bm25_hits:
        return []

    if any(distance is None for _, _, _, distance in dense_hits):
        if not _WARNED_NO_DISTANCE:
            print(
                "  [Vector] WARNING: Some dense hits have no distance; "
                "semantic display score is set to 0."
            )
            _WARNED_NO_DISTANCE = True

    candidates = build_hybrid_candidates(dense_hits, bm25_hits)

    MAX_RERANK_CANDIDATES = 25 if is_aggregation_query(query) else 40
    candidates = candidates[:MAX_RERANK_CANDIDATES]

    if enable_rerank and candidates:
        ranked = rerank_candidates(query, candidates)
    else:
        for item in candidates:
            item["final_score"] = apply_year_penalty(
                query,
                item["content"],
                item["hybrid_score"],
            )
        ranked = sorted(
            candidates,
            key=lambda item: item["final_score"],
            reverse=True,
        )

    if is_aggregation_query(query):
        ranked = sorted(candidates, key=lambda item: item["final_score"], reverse=True)
        return ranked[:10]

    filtered = [
        item for item in ranked 
        if item["final_score"] >= (0.25 if min_score is None else min_score)
    ]
    return filtered[:effective_top_k]


def build_retrieval_verdict(
    results: List[Dict[str, Any]],
    threshold: float = RETRIEVAL_THRESHOLD,
) -> Dict[str, Any]:
    if not results:
        return {
            "contains_answer": False,
            "message": "The answer is not found in the retrieval index.",
            "best_score": 0.0,
        }

    best_score = float(results[0].get("final_score", 0.0))
    contains_answer = best_score >= threshold

    return {
        "contains_answer": contains_answer,
        "message": (
            "The answer is contained in the retrieval index."
            if contains_answer
            else "The answer is not found in the retrieval index."
        ),
        "best_score": best_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline 1: Hybrid Dense + BM25 + RRF + Cross-Encoder"
    )
    parser.add_argument("query", nargs="*", help="Question to ask")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable Cross-Encoder reranking",
    )
    args = parser.parse_args()

    query = " ".join(args.query) if args.query else input("Enter your question: ")
    results = hybrid_search(
        query,
        enable_rerank=not args.no_rerank,
    )

    verdict = build_retrieval_verdict(results)

    print(f"\nQuery: {query}")
    print(verdict["message"])
    print(f"Best final score: {verdict['best_score']:.4f}")
    print(f"Returned chunks: {len(results)}\n")

    for i, result in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Final score:       {result.get('final_score', 0.0):.4f}")
        print(f"RRF score:         {result.get('rrf_score', 0.0):.4f}")
        print(f"BM25 score:        {result.get('bm25_score', 0.0):.4f}")
        print(f"Dense rank:        {result.get('dense_rank')}")
        print(f"BM25 rank:         {result.get('bm25_rank')}")
        if "rerank_score" in result:
            print(f"Cross-Encoder:     {result['rerank_score']:.4f}")
        print(f"Source:            {result['metadata'].get('source')}")
        print(f"Chunk ID:           {result['metadata'].get('chunk_id')}")
        print(f"Content:           {result['content'][:500]}...\n")


if __name__ == "__main__":
    main()