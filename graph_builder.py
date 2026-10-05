import argparse
import collections
import concurrent.futures
import json
import os
import re
import string
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from tg_connection import get_tg_connection
from isection_pipeline import load_documents, split_documents, CHUNK_SIZE, CHUNK_OVERLAP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(BASE_DIR, "graph_build_checkpoint.txt")

# --- Structured Pydantic Models ---
class Relationship(BaseModel):
    source_entity: str = Field(..., description="Name of source entity.")
    target_entity: str = Field(..., description="Name of target entity.")
    relationship: str = Field(..., description="Relationship predicate.")

class EventMetric(BaseModel):
    event_name: str = Field(..., description="Full canonical event name including sport and year.")
    competitors: int = Field(default=0, description="Total individual competitors.")
    nations: int = Field(default=0, description="Total participating nations.")
    year: int = Field(default=0, description="4-digit Olympic year.")

class GraphExtraction(BaseModel):
    relationships: List[Relationship] = Field(default_factory=list)
    metrics: List[EventMetric] = Field(default_factory=list)

# --- Dual Extraction Prompt ---
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert knowledge graph extractor for Olympic sports history.\n\n"
        "RULES:\n"
        "1. RELATIONSHIPS: Extract up to 10 relationships using ONLY: "
        "gold_medal, silver_medal, bronze_medal, competed_in, hosted_at, held_on_date, represented_nation.\n"
        "2. MEDALS: Link medalists directly to full event names: [Athlete] --(gold_medal)--> [Full Event Name].\n"
        "3. METRICS: Extract competitors count, nations count, and year for events into 'metrics'. "
        "Convert word numbers to integers ('forty-seven' -> 47).\n"
        "4. Name events in full (e.g. 'Athletics at the 2008 Summer Olympics – Men\'s 110 metres hurdles').\n\n"
        "Return ONLY a JSON object:\n"
        '{\n'
        '  "relationships": [{"source_entity": "...", "target_entity": "...", "relationship": "..."}],\n'
        '  "metrics": [{"event_name": "...", "competitors": 0, "nations": 0, "year": 0}]\n'
        '}'
    ),
    ("human", "Document title: {title}\n\nContent:\n{text}")
])

def _clean_entity(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return string.capwords(name) if name.islower() else name

def extract_graph_data(text_chunk: str, title: str, llm) -> Optional[GraphExtraction]:
    chain = EXTRACTION_PROMPT | llm
    try:
        msg = chain.invoke({"text": text_chunk, "title": title})
        match = re.search(r"\{.*\}", msg.content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return GraphExtraction(**data)
    except Exception as e:
        pass
    return None

def merge_to_tigergraph(conn, extraction: GraphExtraction, source_file: str, chunk_id: str):
    if not extraction:
        return 0

    count = 0
    # 1. Upsert Relationships
    for t in extraction.relationships:
        src = _clean_entity(t.source_entity)
        tgt = _clean_entity(t.target_entity)
        rel = re.sub(r"\s+", "_", t.relationship.strip().lower())

        if src and tgt and rel:
            conn.upsertVertex("Entity", src, {})
            conn.upsertVertex("Entity", tgt, {})
            conn.upsertEdge("Entity", src, "Relation", "Entity", tgt, {
                "relation": rel,
                "source_file": source_file,
                "chunk_id": chunk_id,
            })
            count += 1

    # 2. Upsert Metrics onto Event Vertices
    for m in extraction.metrics:
        evt = _clean_entity(m.event_name)
        if not evt:
            continue
        attrs = {}
        if m.competitors > 0: attrs["competitors"] = m.competitors
        if m.nations > 0: attrs["nations"] = m.nations
        if m.year > 0: attrs["year"] = m.year

        if attrs:
            conn.upsertVertex("Entity", evt, attrs)

    return count

def _load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE) as f:
        return {int(x.strip()) for x in f if x.strip().isdigit()}

def _save_checkpoint(idx: int):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(f"{idx}\n")

def main():
    parser = argparse.ArgumentParser(description="Parallel Graph Construction for Kaggle Dual T4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    conn = get_tg_connection()
    docs_path = os.path.join(BASE_DIR, "docs")
    documents = load_documents(docs_path)
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=150)

    if args.limit:
        chunks = chunks[:args.limit]

    completed = _load_checkpoint()
    print(f"🚀 Found {len(chunks)} chunks (1000-char / 150-overlap). {len(completed)} already in checkpoint.")

    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0, num_ctx=2048)

    start_time = time.time()
    total_triples = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for idx, chunk in enumerate(chunks, start=1):
            if idx in completed:
                continue
            title = chunk.metadata.get("title", "")
            futures[executor.submit(extract_graph_data, chunk.page_content, title, llm)] = (idx, chunk)

        for future in concurrent.futures.as_completed(futures):
            idx, chunk = futures[future]
            try:
                data = future.result()
                if data:
                    merged = merge_to_tigergraph(
                        conn, data, 
                        chunk.metadata.get("source", "unknown"), 
                        chunk.metadata.get("chunk_id", "unknown")
                    )
                    total_triples += merged
                _save_checkpoint(idx)
                if idx % 100 == 0:
                    print(f"  💾 Processed {idx}/{len(chunks)} chunks | Triples added: {total_triples} | Elapsed: {time.time()-start_time:.0f}s")
            except Exception as e:
                print(f"  ⚠️️ Error processing chunk {idx}: {e}")

    print(f"\n✅ Graph Build Complete! Total triples added: {total_triples} in {time.time()-start_time:.0f}s.")

if __name__ == "__main__":
    main()