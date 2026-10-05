"""Copy the existing Chroma store (chunks + text + embeddings) into TigerGraph `Chunk` vertices.

No re-embedding: vectors are read straight out of Chroma. Run tg_vector_setup.gsql in Savanna first.

    python migrate_chroma_to_tigergraph.py --limit 200 --verify     # small trial first
    python migrate_chroma_to_tigergraph.py --verify                 # full store (resumable via --start)
"""
import argparse
import os
import time

import chromadb

from tg_connection import get_tg_connection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def open_collection(chroma_dir, name):
    if not os.path.isabs(chroma_dir):
        chroma_dir = os.path.join(BASE_DIR, chroma_dir)
    client = chromadb.PersistentClient(path=chroma_dir)
    names = [getattr(c, "name", c) for c in client.list_collections()]
    if name is None:
        if len(names) != 1:
            raise SystemExit(f"Found collections {names}; pass --collection.")
        name = names[0]
    return client.get_collection(name)


def to_float_list(vec):
    return [float(x) for x in vec]


def migrate(col, conn, batch, start, limit):
    total = col.count()
    stop = total if limit is None else min(total, start + limit)
    print(f"Chroma collection has {total} chunks; migrating [{start}:{stop}] in batches of {batch}")

    done = skipped = 0
    t0 = time.time()
    offset = start
    while offset < stop:
        n = min(batch, stop - offset)
        got = col.get(limit=n, offset=offset, include=["documents", "metadatas", "embeddings"])
        ids, docs, metas, embs = got["ids"], got["documents"], got["metadatas"], got["embeddings"]

        rows = []
        for i, cid in enumerate(ids):
            meta = metas[i] or {}
            chunk_id = meta.get("chunk_id") or cid
            if embs is None or embs[i] is None or docs[i] is None:
                skipped += 1
                continue
            rows.append((chunk_id, {
                "content": docs[i],
                "source": str(meta.get("source", "unknown")),
                "emb": to_float_list(embs[i]),
            }))

        if rows:
            accepted = conn.upsertVertices("Chunk", rows)
            if accepted != len(rows):
                print(f"  WARNING: sent {len(rows)} but TigerGraph accepted {accepted}")
            done += accepted
        offset += n
        print(f"  {offset - start}/{stop - start} processed, {done} upserted, {skipped} skipped "
              f"({time.time() - t0:.0f}s elapsed)")

    print(f"\nDone: {done} chunks upserted, {skipped} skipped. Resume point for --start: {offset}")


def verify(col, conn, attempts=6, wait=10):
    """Search with a stored chunk's own embedding: it should come back as the top hit with distance ~0."""
    got = col.get(limit=1, include=["embeddings", "metadatas"])
    meta = got["metadatas"][0] or {}
    cid = meta.get("chunk_id") or got["ids"][0]
    vec = to_float_list(got["embeddings"][0])

    for attempt in range(1, attempts + 1):
        res = conn.runInstalledQuery("chunk_search", {"query_vector": vec, "k": 3})
        print(f"\nverify attempt {attempt}: raw result:\n{res}")
        if res and any(res):
            break
        print(f"  empty; the vector index may still be building. Waiting {wait}s...")
        time.sleep(wait)
    print(f"\nExpected top hit: {cid} with distance close to 0 (if distances look like ~1.0 for the same "
          f"vector, they are similarities, not distances; tell me).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chroma-dir", default="db/chroma_db")
    p.add_argument("--collection", default=None, help="Chroma collection name (auto-detected if only one).")
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--start", type=int, default=0, help="Offset to resume from.")
    p.add_argument("--limit", type=int, default=None, help="Only migrate this many chunks (trial run).")
    p.add_argument("--verify", action="store_true", help="After migrating, run one self-search sanity check.")
    args = p.parse_args()

    col = open_collection(args.chroma_dir, args.collection)
    conn = get_tg_connection()
    migrate(col, conn, args.batch, args.start, args.limit)
    if args.verify:
        verify(col, conn)


if __name__ == "__main__":
    main()