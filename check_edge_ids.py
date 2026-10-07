"""check_edge_ids.py - do Relation-edge chunk_ids exist in the local BM25 index metadata?
Run from the project folder (needs bm25_index.pkl). Read-only, fast (no bulk Chunk fetch)."""
import pickle
from collections import Counter
from tg_connection import get_tg_connection

meta = pickle.load(open("bm25_index.pkl", "rb"))["metadata"]
print("BM25 metadata entries:", len(meta), "| keys:", list(meta[0].keys()))
print("sample BM25 chunk_ids:", [m.get("chunk_id") for m in meta[:3]])
bm_ids = {str(m.get("chunk_id")) for m in meta}
bm_suffix = {i.split("::")[-1] for i in bm_ids}

conn = get_tg_connection()
ids = []
for e in conn.getVertices("Entity", limit=60):
    for ed in conn.getEdges("Entity", e["v_id"], limit=300):
        a = (ed.get("attributes") or {}) if isinstance(ed, dict) else {}
        if a.get("chunk_id"):
            ids.append(str(a["chunk_id"]))
ids = list(dict.fromkeys(ids))
print(f"edge chunk_ids sampled: {len(ids)}")
print("  in BM25 metadata (exact):  ", sum(i in bm_ids for i in ids))
print("  in BM25 metadata (suffix): ", sum(i.split('::')[-1] in bm_suffix for i in ids))
print("  prefix counts:", Counter(i.split('::')[0] for i in ids).most_common(3))
