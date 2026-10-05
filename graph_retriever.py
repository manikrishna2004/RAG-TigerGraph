import argparse
import os
import pickle
import re
from collections import defaultdict
from tg_connection import get_tg_connection

VERTEX_TYPE = "Entity"
CHUNK_TYPE = "Chunk"
EDGE_LIMIT_PER_NODE = 500
MAX_SEED_CANDIDATES = 30
MAX_EXPAND_NODES = 25

_CONN = None
_NODE_NAMES = None
_CACHED_INDEX = None
_EDGE_CACHE = {}
_CHUNK_ATTRS_PRINTED = False
_BM25_MAP = None

_STOPWORDS = {
    "what", "who", "whom", "whose", "which", "when", "where", "why", "how", "did", "does", "do",
    "is", "are", "was", "were", "the", "and", "for", "with", "from", "that", "this", "these",
    "those", "tell", "list", "name", "give", "show", "explain", "describe", "compare", "won",
    "win", "has", "have", "had", "can", "could", "would", "should", "about", "between",
}
_GLUE = {"at", "the", "of", "in", "and", "for", "a", "an", "on", "to"}


def _conn():
    global _CONN
    if _CONN is None:
        _CONN = get_tg_connection()
    return _CONN


def _get_node_names():
    global _NODE_NAMES
    if _NODE_NAMES is None:
        conn = _conn()
        count = conn.getVertexCount(VERTEX_TYPE)
        verts = conn.getVertices(VERTEX_TYPE, limit=max(int(count), 1))
        _NODE_NAMES = [v["v_id"] for v in verts]
        print(f"  [Graph] Loaded {len(_NODE_NAMES)} entity names from TigerGraph")
    return _NODE_NAMES


def _norm(text: str) -> str:
    text = re.sub(r"['’`]", "", str(text).lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def _toks(text: str):
    return [_stem(t) for t in _norm(text).split()]


def _get_index():
    global _CACHED_INDEX
    if _CACHED_INDEX is None:
        _CACHED_INDEX = [(n, _norm(n), set(_toks(n))) for n in _get_node_names()]
    return _CACHED_INDEX


def _match_nodes(text: str) -> list[str]:
    t_norm = f" {_norm(text)} "
    t_toks = set(_toks(text))
    hits = []

    for node, nn, nt in _get_index():
        if len(nn) >= 3 and nn not in _STOPWORDS and f" {nn} " in t_norm:
            hits.append(node)
            continue
        if len(nt) < 4:
            continue
        digits = {t for t in nt if t.isdigit()}
        if not digits or not digits <= t_toks:
            continue
        content = nt - _GLUE
        if not content:
            continue
        missing = len(content - t_toks)
        if missing == 0 or (len(content) >= 7 and missing <= 1):
            hits.append(node)

    # 1. Deduplicate hits contained in longer hits
    hits = [
        h for h in hits 
        if not any(h != o and f" {_norm(h)} " in f" {_norm(o)} " for o in hits)
    ]

    # 2. Filter out generic single-word hits when specific multi-word ones exist
    if any(len(_norm(h).split()) >= 3 for h in hits):
        hits = [h for h in hits if len(_norm(h).split()) >= 2]

    # 3. SPORT-SPECIFIC PRIORITIZATION
    known_sports = [
        "biathlon", "shooting", "athletics", "swimming", "skiing", 
        "cycling", "rowing", "boxing", "gymnastics", "weightlifting",
        "figure skating", "speed skating", "cross country", "curling", "judo"
    ]
    
    # Check if text or any matched entity mentions a known sport
    matched_sports = []
    for s in known_sports:
        if f" {s} " in t_norm:
            matched_sports.append(s)
        else:
            for hit in hits:
                if s in _norm(hit):
                    matched_sports.append(s)
                    break

    if matched_sports:
        sport_specific_hits = [
            hit for hit in hits 
            if any(s in _norm(hit) for s in matched_sports)
        ]
        if sport_specific_hits:
            hits = sport_specific_hits

    return hits


def find_entities_in_question(question: str, max_entities: int = 8):
    hits = _match_nodes(question)

    if not hits:
        from difflib import SequenceMatcher
        q_tokens = [t for t in _norm(question).split() if t not in _STOPWORDS and len(t) > 4]
        for node, nn, _ in _get_index():
            if any(SequenceMatcher(None, nn, t).ratio() > 0.9 for t in q_tokens):
                hits.append(node)
    if not hits:
        tokens = {t.lower() for t in re.findall(r"\b[A-Z][A-Za-z0-9'\-]{2,}\b", question)} - _STOPWORDS
        if tokens:
            for node, nn, _ in _get_index():
                if tokens & set(nn.split()):
                    hits.append(node)

    hits = hits[:MAX_SEED_CANDIDATES]

    def score(name):
        nn = _norm(name)
        return len(nn.split()) * 10.0 + len(nn) * 0.5 + min(len(_edges_of(name)), 50)

    hits.sort(key=score, reverse=True)
    return hits[:max_entities]


def _edge_record(u, v, data):
    relation = data.get("relation", "connected_to")
    return {
        "source": u, "target": v, "relation": relation,
        "source_file": data.get("source_file", "unknown"),
        "chunk_id": data.get("chunk_id", "unknown"),
        "fact_str": f"{u} --[{relation}]--> {v}",
    }


def _edges_of(node):
    if node in _EDGE_CACHE:
        return _EDGE_CACHE[node]
    records = []
    try:
        edges = _conn().getEdges(VERTEX_TYPE, node, limit=EDGE_LIMIT_PER_NODE)
    except Exception as e:
        print(f"  [Graph] Edge fetch failed for '{node}': {e}")
        edges = []
    for e in edges:
        etype = e.get("e_type", "")
        if "relation" not in etype.lower():
            continue
        u, v = e.get("from_id"), e.get("to_id")
        if etype.lower().startswith("reverse"):
            u, v = v, u
        records.append(_edge_record(u, v, e.get("attributes", {})))
    _EDGE_CACHE[node] = records
    return records


def _chunk_text(attrs: dict) -> str:
    if isinstance(attrs.get("content"), str) and attrs["content"].strip():
        return attrs["content"]
    skip = {"source", "source_file", "chunk_id", "doc_id", "url", "title", "id"}
    best = ""
    for k, val in attrs.items():
        if isinstance(val, str) and k.lower() not in skip and len(val) > len(best):
            best = val
    return best


_BM25_PAYLOAD = None
_BM25_OBJ = None


def _bm25_payload():
    global _BM25_PAYLOAD
    if _BM25_PAYLOAD is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm25_index.pkl")
        try:
            with open(path, "rb") as f:
                _BM25_PAYLOAD = pickle.load(f)
            print(f"  [Graph] Loaded corpus text index ({len(_BM25_PAYLOAD['metadata'])} chunks) from bm25_index.pkl")
        except Exception as e:
            print(f"  [Graph] corpus text index unavailable: {e}")
            _BM25_PAYLOAD = {}
    return _BM25_PAYLOAD


def _bm25_chunk_map():
    global _BM25_MAP
    if _BM25_MAP is None:
        p = _bm25_payload()
        _BM25_MAP = {m["chunk_id"]: m["content"] for m in p.get("metadata", []) if m.get("chunk_id")}
    return _BM25_MAP


def _resolve_by_overlap(recs, seeds):
    global _BM25_OBJ
    p = _bm25_payload()
    if not p:
        return None
    if _BM25_OBJ is None:
        from rank_bm25 import BM25Okapi
        _BM25_OBJ = BM25Okapi(p["tokenized_corpus"])
    import numpy as np
    seed_set = set(seeds)
    names = []
    for r in recs:
        for e in (r["source"], r["target"]):
            if e not in seed_set and len(_norm(e)) >= 4:
                names.append(e)
    names = list(dict.fromkeys(names))[:6]
    if not names:
        return None
    ev_toks = [t for t in _norm(seeds[0]).split() if t not in _GLUE] if seeds else []
    digits = {t for t in ev_toks if t.isdigit()}
    q = [t for n in names for t in _norm(n).split()] + ev_toks
    scores = _BM25_OBJ.get_scores(q)
    meta = p["metadata"]
    nn = [_norm(n) for n in names]
    for i in np.argsort(scores)[::-1][:8]:
        text = meta[int(i)]["content"]
        tn = f" {_norm(text)} "
        if digits and not (digits & set(tn.split())):
            continue
        if any(f" {n} " in tn or n in tn for n in nn):
            return meta[int(i)]["chunk_id"], text
    return None


# In graph_retriever.py

def _fetch_chunks(ids):
    global _CHUNK_ATTRS_PRINTED
    out = {}
    try:
        verts = _conn().getVerticesById(CHUNK_TYPE, list(ids))
        if isinstance(verts, dict):
            verts = [verts]
        for v in verts or []:
            attrs = v.get("attributes", {})
            text = _chunk_text(attrs)
            if text:
                out[v.get("v_id")] = text
    except Exception:
        # Silently retry ID by ID or pass over Kaggle-generated ID mismatches
        for i in ids:
            try:
                vs = _conn().getVerticesById(CHUNK_TYPE, [i])
                if isinstance(vs, dict):
                    vs = [vs]
                for v in vs or []:
                    text = _chunk_text(v.get("attributes", {}))
                    if text:
                        out[v.get("v_id")] = text
            except Exception:
                pass

    missing = [i for i in ids if i not in out]
    if missing:
        m = _bm25_chunk_map()
        for i in missing:
            if i in m:
                out[i] = m[i]
    return out

def graph_chunk_search(question: str, max_chunks: int = 4, seeds=None):
    try:
        seeds = seeds or find_entities_in_question(question)
    except Exception as e:
        print(f"  [Graph] TigerGraph lookup failed: {e}")
        return []
    if not seeds:
        print(f"  [Graph] No entities matched: '{question}'")
        return []
    print(f"  [Graph] Matched entities: {seeds}")

    score, via, recs_by = defaultdict(float), defaultdict(set), defaultdict(list)
    for rank, s in enumerate(seeds):
        w = 1.0 / (1.0 + 0.5 * rank)
        for r in _edges_of(s):
            cid = r["chunk_id"]
            if cid and cid != "unknown":
                score[cid] += w
                via[cid].add(s)
                recs_by[cid].append(r)
    if not score:
        return []

    cand = sorted(score, key=score.get, reverse=True)[:min(len(score), 15)]
    texts = _fetch_chunks(cand)
    n_by_id = len(texts)
    for cid in [c for c in cand if c not in texts]:
        hit = _resolve_by_overlap(recs_by[cid], seeds)
        if hit:
            local_id, text = hit
            if local_id not in texts:
                texts[local_id] = text
                score[local_id] = score[cid]
                via[local_id] = via[cid]
    print(f"  [Graph] passages: {n_by_id} by chunk_id, {len(texts) - n_by_id} by entity-overlap lookup "
          f"(of {len(cand)} graph chunk_ids)")
    if not texts:
        return []

    q_content = {t for t in _toks(question) if t not in _GLUE and t not in _STOPWORDS}
    top = max(score[c] for c in texts)
    ranked = []
    for cid, text in texts.items():
        overlap = len(q_content & set(_toks(text))) / max(len(q_content), 1)
        ranked.append((score[cid] / top + overlap, cid, text))
    ranked.sort(reverse=True)
    return [{"chunk_id": cid, "content": text, "graph_score": round(sc, 3), "seeds": sorted(via[cid])}
            for sc, cid, text in ranked[:max_chunks]]


def entity_attributes(nodes):
    out = {}
    if not nodes:
        return out
    try:
        verts = _conn().getVerticesById(VERTEX_TYPE, list(nodes))
    except Exception as e:
        print(f"  [Graph] Entity attribute fetch failed: {e}")
        return out
    if isinstance(verts, dict):
        verts = [verts]
    for v in verts or []:
        a = {k: val for k, val in (v.get("attributes") or {}).items()
             if val not in (0, None, "", [], {}) and not isinstance(val, (list, dict))}
        if a:
            out[v.get("v_id")] = a
    return out


def attributes_text(nodes):
    attrs = entity_attributes(nodes)
    return "\n".join(f"{n}: " + ", ".join(f"{k}={v}" for k, v in a.items()) for n, a in attrs.items())


def _expand(seeds, max_depth: int, max_facts: int, question: str = None):
    qn = set(_norm(question).split()) if question else set()
    q_tokens = {t for t in qn if len(t) > 2 and t not in _STOPWORDS}
    seed_set = set(seeds)

    def score(rec):
        return len(q_tokens & set(_norm(rec["fact_str"]).split()))

    seen, kept = set(), []

    def take(recs, limit):
        new = []
        for r in sorted(recs, key=score, reverse=True):
            key = (r["source"], r["target"], r["relation"])
            if key in seen:
                continue
            seen.add(key)
            new.append(r)
            if len(new) >= limit:
                break
        kept.extend(new)
        return new

    reach, links = {}, {}
    for s in seeds:
        for r in _edges_of(s):
            o = r["target"] if r["source"] == s else r["source"]
            if o in seed_set:
                continue
            reach.setdefault(o, set()).add(s)
            links.setdefault(o, []).append(r)
    joins = [n for n, ss in reach.items() if len(ss) >= 2]
    joins.sort(key=lambda n: (-len(reach[n]), len(_edges_of(n))))
    joins = joins[:5]
    visited = set(seeds) | set(joins)

    if joins:
        per = max(8, (max_facts * 3 // 5) // len(joins))
        for n in joins:
            take(links[n], 2)
            take(_edges_of(n), per - 2)

    remaining = max_facts - len(kept)
    hop1_budget = remaining if max_depth <= 1 else remaining // 2
    per_seed = max(3, hop1_budget // max(len(seeds), 1))
    cand = []
    for s in seeds:
        for r in take(_edges_of(s), per_seed):
            o = r["target"] if r["source"] == s else r["source"]
            if o not in visited:
                visited.add(o)
                cand.append((score(r), o))
    frontier = [n for _, n in sorted(cand, key=lambda x: -x[0])]

    for _ in range(1, max_depth):
        if not frontier or len(kept) >= max_facts:
            break
        pool = []
        for node in frontier[:MAX_EXPAND_NODES]:
            pool.extend(_edges_of(node))
        new = take(pool, max_facts - len(kept))
        frontier = []
        for r in new:
            for o in (r["source"], r["target"]):
                if o not in visited:
                    visited.add(o)
                    frontier.append(o)
    return kept[:max_facts]


def search_question_subgraph(question, max_depth=2, graph_path=None, max_facts=60):
    try:
        seeds = find_entities_in_question(question)
    except Exception as e:
        print(f"  [Graph] TigerGraph lookup failed: {e}")
        return []
    if not seeds:
        print(f"  [Graph] No entities from the question found in graph: '{question}'")
        return []
    print(f"  [Graph] Matched entities: {seeds}")
    return _expand(seeds, max_depth, max_facts, question)


def search_entity_subgraph(query_entity: str, max_depth: int = 2, graph_path: str = None, max_facts: int = 60):
    try:
        matching = _match_nodes(query_entity)[:MAX_SEED_CANDIDATES]
    except Exception as e:
        print(f"  [Graph] TigerGraph lookup failed: {e}")
        return []
    if not matching:
        print(f"No direct entity match found for: '{query_entity}'")
        return []
    return _expand(matching, max_depth, max_facts)


def search_entities_subgraph(entities: list, max_depth: int = 2, max_facts: int = 60):
    if not entities:
        return []
    try:
        seeds = []
        for entity in entities:
            if entity:
                m = sorted(_match_nodes(str(entity)), key=lambda n: -len(_norm(n)))
                seeds.extend(m[:3])
    except Exception as e:
        print(f"  [Graph] TigerGraph lookup failed: {e}")
        return []
    seeds = list(dict.fromkeys(seeds))[:MAX_SEED_CANDIDATES]
    if not seeds:
        print(f"  [Graph] No graph entities matched from seed list: {entities}")
        return []
    print(f"  [Graph] Entity-seeded traversal from: {seeds}")
    return _expand(seeds, max_depth, max_facts)


def main():
    parser = argparse.ArgumentParser(description="Query the knowledge graph (TigerGraph)")
    parser.add_argument("entity", nargs="*")
    parser.add_argument("--question", action="store_true")
    parser.add_argument("--chunks", action="store_true", help="show chunk text reached via the graph")
    args = parser.parse_args()
    text = " ".join(args.entity) if args.entity else input("Enter an entity / question: ")

    if args.chunks:
        res = graph_chunk_search(text)
        print(f"\nFound {len(res)} chunks for '{text}':")
        for r in res:
            print(f"\n--- {r['chunk_id']} score={r['graph_score']} via={r['seeds']}\n{r['content'][:700]}")
        return
    results = search_question_subgraph(text) if args.question else search_entity_subgraph(text)
    print(f"\nFound {len(results)} relationships for '{text}':")
    for r in results:
        print(f" - {r['fact_str']}  [chunk: {r['chunk_id']}]")


if __name__ == "__main__":
    main()