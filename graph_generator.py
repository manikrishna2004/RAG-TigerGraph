"""
graph_generator.py — P2: pure GraphRAG (no vector search, no BM25).
question -> entity nodes -> chunk_ids on their edges -> chunk text (graph_chunk_search)
        + entity attributes (competitors / nations / year) -> one LLM call.
Falls back to triple facts only if chunk text can't be resolved.
"""
import argparse
import json
import sys
import textwrap
import urllib.request
import urllib.error
from graph_retriever import graph_chunk_search, search_question_subgraph, attributes_text

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_MAX_FACTS = 50
DEFAULT_DEPTH = 2
MAX_CHUNKS = 4
MAX_CHUNK_CHARS = 1800

SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer ONLY from the retrieved knowledge graph facts below.

STRICT RULES:
1. DISCARD CONFLICTS: If multiple conflicting entities claim the same award (e.g. two gold medalists in a single-winner event), inspect the exact event attributes (year, gender, distance) before accepting them.
2. DO NOT MERGE: Do not assume multiple listed entities won the same medal unless explicitly stated as a tie.
3. INSUFFICIENT EVIDENCE: If facts conflict or do not clearly identify a single winner, state that the graph evidence is inconclusive.
"""

def format_context(chunks: list[dict] = None, question: str = "", facts: list[dict] = None) -> str:
    """Format graph-anchored chunks (or fallback raw triples) into prompt context."""
    if not chunks and not facts:
        return "(no relevant facts or chunks found in the knowledge graph)"

    blocks = []
    
    # 1. Primary path: Format graph-anchored document passage chunks
    if chunks:
        for i, c in enumerate(chunks, 1):
            cid = c.get("chunk_id", "unknown")
            via = c.get("seeds", [])
            content = c.get("content", "")
            blocks.append(f"[Doc {i}] (Chunk: {cid} | via graph entities: {via})\n{content}")
        return "\n\n".join(blocks)

    # 2. Fallback path: Format raw triples if no passage chunks resolved
    lines = []
    for i, rec in enumerate(facts or [], 1):
        line = f"{i:3}. {rec['fact_str']}"
        if rec.get("source_file") and rec["source_file"] != "unknown":
            line += f"  [src: {rec['source_file']}]"
        lines.append(line)
    return "\n".join(lines)

def call_ollama(prompt, model, num_ctx=6144):
    payload = json.dumps({
        "model": model, "prompt": prompt, "system": SYSTEM_PROMPT, "stream": False,
        "keep_alive": "30m",
        "options": {"num_ctx": num_ctx, "temperature": 0, "num_predict": 300},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            d = json.loads(resp.read())
            return d.get("response", "").strip(), int(d.get("prompt_eval_count", 0)) + int(d.get("eval_count", 0))
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Could not reach Ollama at {OLLAMA_URL}: {e}")
        sys.exit(1)


def answer_question(question, model=DEFAULT_MODEL, max_facts=DEFAULT_MAX_FACTS, depth=DEFAULT_DEPTH,
                    dry_run=False, verbose=False) -> dict:
    print(f"\n[1/3] Graph retrieval for: '{question}'")
    chunks = graph_chunk_search(question, max_chunks=MAX_CHUNKS)
    seeds = sorted({s for c in chunks for s in c["seeds"]}) if chunks else []
    facts = []
    if not chunks:   # chunk text unresolved -> triples as a weak fallback
        facts = search_question_subgraph(question, max_depth=depth, max_facts=max_facts)
        seeds = []
    attrs = attributes_text(seeds[:4]) if seeds else ""
    print(f"      -> {len(chunks)} chunks via graph, {len(facts)} fallback facts")

    context = format_context(chunks=chunks, question=question, facts=facts)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    if dry_run:
        print(textwrap.indent(prompt, "  "))
        return {"answer": "", "facts": chunks if chunks else facts, "chunks": chunks, "context_str": context, "tokens": 0}

    print(f"[3/3] Generating with {model}...")
    answer, tokens = call_ollama(prompt, model)
    return {"answer": answer, "facts": chunks if chunks else facts, "chunks": chunks, "context_str": context, "tokens": tokens}


def main():
    p = argparse.ArgumentParser(description="P2: pure GraphRAG (TigerGraph + Ollama)")
    p.add_argument("question", nargs="*")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    q = " ".join(a.question) if a.question else input("Question: ")
    r = answer_question(q, model=a.model, dry_run=a.dry_run)
    if r["answer"]:
        print("\nANSWER:\n" + textwrap.fill(r["answer"], 80) + f"\n(tokens: {r['tokens']})")


if __name__ == "__main__":
    main()