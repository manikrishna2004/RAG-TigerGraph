"""chunk_resolver.py - map Relation-edge `chunk_id` values to Chunk vertices in TigerGraph.

Why: graph_retriever gets chunk ids from edge attributes, then calls getVerticesById("Chunk", ...)
and fails with 601 "not a valid vertex id". This module builds a one-time alias index over ALL
Chunk vertices (v_id, id suffix after '::', every short string attribute, and 12-hex md5/sha1/sha256
of the content) and resolves edge ids through it.

Run `python chunk_resolver.py` FIRST: it prints what your graph actually contains and which
strategy resolves the edge ids. Nothing is assumed about your schema.
"""
import hashlib
from collections import defaultdict
from typing import Dict, List
from tg_connection import get_tg_connection

CHUNK_TYPE, ENTITY_TYPE = "Chunk", "Entity"
_ALIAS: Dict[str, str] = {}      # alias -> chunk v_id (unambiguous aliases only)
_CONTENT: Dict[str, str] = {}    # chunk v_id -> content
_BUILT = False


def _aliases(v: dict) -> set:
    vid = str(v.get("v_id", ""))
    a = v.get("attributes") or {}
    out = {vid, vid.split("::")[-1]}
    for k, val in a.items():
        if k != "content" and isinstance(val, str) and 0 < len(val) <= 80:
            out.add(val)
            out.add(val.split("::")[-1])
    content = str(a.get("content", ""))
    if content:
        for h in (hashlib.md5, hashlib.sha1, hashlib.sha256):
            for txt in (content, content.strip()):
                out.add(h(txt.encode("utf-8")).hexdigest()[:12])
    return {x for x in out if x}


def build_index(force: bool = False) -> None:
    global _BUILT
    if _BUILT and not force:
        return
    conn = get_tg_connection()
    n = int(conn.getVertexCount(CHUNK_TYPE))
    verts = conn.getVertices(CHUNK_TYPE, limit=max(n, 1))
    owners = defaultdict(set)
    for v in verts:
        vid = str(v.get("v_id", ""))
        _CONTENT[vid] = str((v.get("attributes") or {}).get("content", ""))
        for al in _aliases(v):
            owners[al].add(vid)
    for al, vids in owners.items():
        if len(vids) == 1:                       # drop ambiguous aliases (e.g. 'corpus.jsonl')
            _ALIAS[al] = next(iter(vids))
    _BUILT = True


def resolve(ids: List[str]) -> Dict[str, str]:
    """{requested_id: chunk_text} for every id that can be resolved."""
    build_index()
    out = {}
    for cid in ids:
        for key in (cid, cid.split("::")[-1], "corpus.jsonl::" + cid.split("::")[-1]):
            vid = _ALIAS.get(key)
            if vid and _CONTENT.get(vid):
                out[cid] = _CONTENT[vid]
                break
    return out


# ------------------------------------------------------------------ diagnostics
def _main():
    conn = get_tg_connection()
    n = int(conn.getVertexCount(CHUNK_TYPE))
    print(f"Chunk vertices: {n}")
    sample = conn.getVertices(CHUNK_TYPE, limit=3)
    for v in sample:
        a = v.get("attributes") or {}
        print(f"  Chunk v_id={v.get('v_id')!r} attrs={list(a.keys())} "
              f"content[:60]={str(a.get('content',''))[:60]!r}")

    ents = conn.getVertices(ENTITY_TYPE, limit=60)
    etypes, edge_ids, shown = defaultdict(int), [], 0
    for e in ents:
        try:
            edges = conn.getEdges(ENTITY_TYPE, e["v_id"], limit=300)
        except Exception as ex:
            print("  getEdges failed:", ex); continue
        for ed in edges:
            if not isinstance(ed, dict):
                continue
            et = str(ed.get("e_type", ""))
            etypes[et] += 1
            at = ed.get("attributes") or {}
            if shown < 3 and "relation" in et.lower():
                print(f"  edge e_type={et!r} attr_keys={list(at.keys())} chunk_id={at.get('chunk_id')!r}")
                shown += 1
            if at.get("chunk_id"):
                edge_ids.append(str(at["chunk_id"]))
    print("Edge types seen (count):", dict(etypes))
    print("  -> graph_retriever keeps only types containing 'relation':",
          [t for t in etypes if "relation" in t.lower()] or "NONE (every edge would be dropped!)")
    edge_ids = list(dict.fromkeys(edge_ids))[:300]
    print(f"Distinct edge chunk_ids sampled: {len(edge_ids)}  e.g. {edge_ids[:3]}")
    if not edge_ids:
        return

    ok_exact = 0
    for cid in edge_ids[:30]:
        try:
            if conn.getVerticesById(CHUNK_TYPE, cid):
                ok_exact += 1
        except Exception:
            pass
    print(f"Direct getVerticesById (one id at a time) hits: {ok_exact}/{min(30, len(edge_ids))}")

    build_index()
    hit = resolve(edge_ids)
    print(f"Alias-index resolution: {len(hit)}/{len(edge_ids)} resolved")
    miss = [c for c in edge_ids if c not in hit][:5]
    if miss:
        print("  unresolved examples:", miss)
    print("Chunk v_id examples:", list(_CONTENT)[:3])


if __name__ == "__main__":
    _main()
