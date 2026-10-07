import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from sentence_transformers import CrossEncoder
from tg_connection import get_tg_connection

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VERTEX_TYPE = "Entity"
CHUNK_TYPE = "Chunk"
EDGE_LIMIT_PER_NODE = 300        
MAX_SEED_ENTITIES = 7           
MAX_CANDIDATE_CHUNKS = 25        
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

_CONN = None
_NODE_NAMES = None
_RERANKER = None
_EDGE_CACHE = {}

# Expanded stopword list to prevent generic matches grabbing the whole graph
_GLUE = {"at", "the", "of", "in", "and", "for", "a", "an", "on", "to", "by", "with", "who", "what", "where", "when", "how", "many", "which"}


def _conn():
    global _CONN
    if _CONN is None:
        _CONN = get_tg_connection()
    return _CONN


def get_reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(4)
        print(f"  [Pure GraphRAG] Loading Reranker '{RERANKER_MODEL_NAME}' on {device.upper()}...")
        _RERANKER = CrossEncoder(RERANKER_MODEL_NAME, max_length=256, device=device)
    return _RERANKER


def _get_node_names() -> List[str]:
    global _NODE_NAMES
    if _NODE_NAMES is None:
        conn = _conn()
        try:
            count = conn.getVertexCount(VERTEX_TYPE)
            verts = conn.getVertices(VERTEX_TYPE, limit=max(int(count), 1))
            _NODE_NAMES = [v["v_id"] for v in verts]
        except Exception as e:
            print(f"  [Pure GraphRAG Error] Failed to fetch entity vertices: {e}")
            _NODE_NAMES = []
    return _NODE_NAMES


def _norm(text: str) -> str:
    text = re.sub(r"['’`]", "", str(text).lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


# ---------------------------------------------------------------------------
# Aggressive Entity Linking 
# ---------------------------------------------------------------------------
def find_entities_in_question(question: str, max_entities: int = MAX_SEED_ENTITIES) -> List[str]:
    names = _get_node_names()
    if not names:
        return []

    q_norm = f" {_norm(question)} "
    q_toks = set(_norm(question).split()) - _GLUE

    scored_matches = []
    for node in names:
        n_norm = _norm(node)
        if not n_norm or len(n_norm) < 2:
            continue
        
        # 1. Exact Substring Boost
        if f" {n_norm} " in q_norm:
            scored_matches.append((node, 10.0 + len(n_norm)))
            continue
            
        # 2. Token Overlap Scoring
        n_toks = set(n_norm.split()) - _GLUE
        if not n_toks:
            continue
            
        overlap = len(n_toks & q_toks)
        overlap_ratio = overlap / len(n_toks)
        
        if overlap_ratio >= 0.4:
            scored_matches.append((node, overlap_ratio * len(n_norm)))

    scored_matches.sort(key=lambda x: x[1], reverse=True)
    
    seen = set()
    final_seeds = []
    for node, _ in scored_matches:
        if node not in seen:
            seen.add(node)
            final_seeds.append(node)
            
    return final_seeds[:max_entities]


# ---------------------------------------------------------------------------
# Traversal & Fetching
# ---------------------------------------------------------------------------
def _edges_of(node: str) -> List[Dict[str, Any]]:
    if node in _EDGE_CACHE:
        return _EDGE_CACHE[node]
    records = []
    try:
        edges = _conn().getEdges(VERTEX_TYPE, node, limit=EDGE_LIMIT_PER_NODE)
    except Exception:
        edges = []

    for e in edges:
        if not isinstance(e, dict):
            continue

        etype = str(e.get("e_type", ""))
        if "relation" not in etype.lower():
            continue
        
        u, v = e.get("from_id"), e.get("to_id")
        if etype.lower().startswith("reverse"):
            u, v = v, u

        attrs = e.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}

        records.append({
            "source": u,
            "target": v,
            "relation": attrs.get("relation", "connected_to"),
            "chunk_id": attrs.get("chunk_id", "unknown"),
        })
    _EDGE_CACHE[node] = records
    return records


def _fetch_graph_chunks(ids: List[str]) -> Dict[str, str]:
    out = {}
    if not ids:
        return out
        
    conn = _conn()
    clean_ids = [cid.split("::")[-1] if "::" in cid else cid for cid in ids]
    
    verts = None
    try:
        verts = conn.getVerticesById(CHUNK_TYPE, ids)
    except Exception:
        pass
        
    if not verts:
        try:
            verts = conn.getVerticesById(CHUNK_TYPE, clean_ids)
        except Exception as e:
            print(f"  [Pure GraphRAG Error] Both ID formats failed validation: {e}")
            verts = []
            
    if isinstance(verts, dict):
        verts = [verts]
        
    for v in (verts or []):
        if not isinstance(v, dict):
            continue
        attrs = v.get("attributes", {})
        if isinstance(attrs, dict):
            content = str(attrs.get("content", ""))
            v_id = str(v.get("v_id", ""))
            
            if content and v_id:
                out[v_id] = content
                out[f"corpus.jsonl::{v_id}"] = content
                
    return out


# ---------------------------------------------------------------------------
# REQUIRED FIX: Entity Attribute Extractor
# ---------------------------------------------------------------------------
def attributes_text(nodes: List[str]) -> str:
    """Extract entity attributes directly from TigerGraph Entity vertices for Context."""
    if not nodes:
        return ""
    out = []
    try:
        verts = _conn().getVerticesById(VERTEX_TYPE, list(nodes))
        if isinstance(verts, dict):
            verts = [verts]
        for v in (verts or []):
            if not isinstance(v, dict):
                continue
            attrs_obj = v.get("attributes")
            attrs = attrs_obj if isinstance(attrs_obj, dict) else {}
            # Filter out empty values and internal IDs
            filtered_attrs = {k: val for k, val in attrs.items() if val and k != "v_id"}
            if filtered_attrs:
                v_id = v.get("v_id")
                out.append(f"{v_id}: " + ", ".join(f"{k}={val}" for k, val in filtered_attrs.items()))
    except Exception:
        pass
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------
def graph_chunk_search(question: str, max_chunks: int = 5, seeds: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    seeds = seeds if seeds is not None else find_entities_in_question(question)
    if not seeds:
        print(f"  [Pure GraphRAG] No entity vertices matched in graph for: '{question}'")
        return []

    print(f"  [Pure GraphRAG] Traversing TigerGraph subgraphs from seed entities: {seeds}")

    score_map, via_map = defaultdict(float), defaultdict(set)
    for rank, s in enumerate(seeds):
        weight = 1.0 / (1.0 + 0.5 * rank)
        for r in _edges_of(s):
            cid = r["chunk_id"]
            if cid and cid != "unknown":
                score_map[cid] += weight
                via_map[cid].add(s)

    if not score_map:
        print(f"  [Pure GraphRAG] No chunk vertices connected to seeds: {seeds}")
        return []

    candidate_ids = sorted(score_map, key=score_map.get, reverse=True)[:MAX_CANDIDATE_CHUNKS]
    texts_map = _fetch_graph_chunks(candidate_ids)

    if not texts_map:
        print(f"  [Pure GraphRAG] No content attributes found on target chunk vertices.")
        return []

    candidates = []
    pairs = []
    cid_list = []

    for cid in candidate_ids:
        if cid in texts_map:
            pairs.append([question, texts_map[cid]])
            cid_list.append(cid)

    if not pairs:
        return []

    reranker = get_reranker()
    raw_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)

    max_graph_score = max(score_map.values()) if score_map.values() else 1.0

    for cid, raw_score in zip(cid_list, raw_scores):
        rerank_score = 1.0 / (1.0 + torch.exp(torch.tensor(-float(raw_score))).item())
        graph_norm = score_map[cid] / max_graph_score
        
        combined_score = (0.60 * rerank_score) + (0.40 * graph_norm)

        candidates.append({
            "chunk_id": cid,
            "content": texts_map[cid],
            "graph_score": round(combined_score, 4),
            "seeds": sorted(via_map[cid])
        })

    candidates.sort(key=lambda x: x["graph_score"], reverse=True)
    return candidates[:max_chunks]