import argparse
import json
import os
import pickle
import re
import time
import numpy as np

from langchain_ollama import ChatOllama, OllamaEmbeddings
from rank_bm25 import BM25Okapi
from tg_connection import get_tg_connection
from rag_generator import generate_answer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BM25_INDEX_PATH = os.path.join(BASE_DIR, "bm25_index.pkl")

# Strict Fact Verification System Prompt to prevent attribute/gender collision
PIPELINE_1_SYSTEM_PROMPT = """You are a precise, evidence-grounded factual assistant. 
Follow these rules strictly when answering from the provided context:
1. ATTRIBUTE VERIFICATION: Verify that the Gender (Men/Women), Event Type (Individual/Team/Relay), Distance, and Year match the user query EXACTLY.
2. If a retrieved document discusses a different gender or a similar variant event (e.g., Women's instead of Men's, or 200m medley instead of 200m freestyle), EXPLICITLY DISCARD IT. Do not substitute or extrapolate.
3. EXHAUSTIVE COUNTING: For counting queries, scan all provided chunks carefully and list every matching event before calculating the final count.
4. If the exact answer cannot be found in the verified chunks, state clearly that the text does not contain it."""


class Pipeline1HybridRetriever:
    def __init__(self):
        print(f"Loading BM25 index from {BM25_INDEX_PATH}...")
        if not os.path.exists(BM25_INDEX_PATH):
            raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}. Run isection_pipeline.py first!")
        
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
            
        self.tokenized_corpus = data["tokenized_corpus"]
        self.metadata = data["metadata"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        self.conn = get_tg_connection()
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    def _vector_search(self, query: str, top_k: int):
        """Execute vector similarity search against TigerGraph using 'chunk_search'."""
        prefixed_query = f"search_query: {query}"
        query_emb = self.embedding_model.embed_query(prefixed_query)
        
        # Convert numpy array or tensor to standard Python list for JSON serialization
        if hasattr(query_emb, "tolist"):
            query_emb = query_emb.tolist()

        try:
            # Calls your exact installed GSQL query 'chunk_search' with parameter 'query_vector'
            result = self.conn.runInstalledQuery(
                "chunk_search", 
                params={"query_vector": query_emb, "k": top_k},
                timeout=30000
            )
            
            chunks = []
            # pyTigerGraph typically returns a list of output maps or dictionaries
            if result:
                # Handle list or dict return types safely
                response_items = result if isinstance(result, list) else [result]
                
                for item in response_items:
                    # Extract vertex sets from PRINT output (typically mapped under 'v' or 'v_set')
                    vertex_list = item.get("v", item.get("v_set", []))
                    for v_item in vertex_list:
                        attr = v_item.get("attributes", {})
                        chunks.append({
                            "chunk_id": v_item.get("v_id"),
                            "content": attr.get("content", ""),
                            "source": attr.get("source", "")
                        })
            return chunks
        except Exception as e:
            print(f"  [Warning] TigerGraph vector search query error: {e}. Falling back to BM25-only.")
            return []

    def retrieve(self, query: str, top_k: int = 5):
        query_lower = query.lower()
        query_tokens = re.findall(r"[a-z0-9]+", query_lower)
        
        # 1. BM25 Lexical Scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # 2. Dense Vector Search Results
        vector_results = self._vector_search(query, top_k=top_k * 2)
        
        # 3. Reciprocal Rank Fusion (RRF)
        fusion_scores = {}
        
        # Add vector ranks
        for rank, doc in enumerate(vector_results):
            doc_id = doc["chunk_id"]
            fusion_scores[doc_id] = {
                "score": fusion_scores.get(doc_id, {}).get("score", 0.0) + 1.0 / (60 + rank),
                "content": doc["content"],
                "source": doc["source"]
            }
            
        # Add BM25 ranks
        for rank, idx in enumerate(top_bm25_indices):
            meta = self.metadata[idx]
            doc_id = meta["chunk_id"]
            fusion_scores[doc_id] = {
                "score": fusion_scores.get(doc_id, {}).get("score", 0.0) + 1.0 / (60 + rank),
                "content": meta["content"],
                "source": meta["source"]
            }
            
        # Sort and return top_k
        sorted_docs = sorted(fusion_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return sorted_docs


def get_dynamic_k(query: str, default_k: int = 5) -> int:
    """Dynamically expand K for aggregation and counting queries."""
    query_lower = query.lower()
    aggregation_triggers = ["how many", "list all", "total events", "highest number", "all events"]
    if any(trigger in query_lower for trigger in aggregation_triggers):
        return 30  # Pull a deeper pool of chunks for multi-event aggregation
    return default_k


import argparse
import json
import os
import time

from rag_generator import generate_answer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import re

def evaluate_answer(generated_answer: str, gold_answers: list) -> bool:
    """Strict evaluation with boundary matching to avoid substring digit collisions."""
    if not gold_answers:
        return False
        
    gen_lower = generated_answer.lower()
    
    for gold in gold_answers:
        gold_str = str(gold).strip().lower()
        
        # If gold answer is a single number (e.g. "4", "20", "47")
        if gold_str.isdigit():
            # Match number with word boundaries (e.g., " 4 ", "is 4.", "total of 4")
            # Prevents matching inside years like "2014" or "20km"
            pattern = r"\b" + re.escape(gold_str) + r"\b"
            if re.search(pattern, gen_lower):
                return True
        else:
            # For textual names like "Renaud Lavillenie" or "Michael Phelps"
            if gold_str in gen_lower:
                return True
                
    return False


def main():
    parser = argparse.ArgumentParser(description="Pipeline 1 (Pure RAG) Benchmark")
    parser.add_argument("--eval_file", type=str, default="eval_public_missing.json", help="Path to evaluation set json")
    args = parser.parse_args()

    eval_path = os.path.join(BASE_DIR, args.eval_file)
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_path}.")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"\n🚀 Running Pipeline 1 Benchmark on {len(eval_data)} queries from {args.eval_file}...\n")

    results_summary = []
    total_matches = 0
    start_time_all = time.time()

    for idx, item in enumerate(eval_data, start=1):
        query = item.get("query") or item.get("question")
        
        # FIX: Explicitly check 'gold' array first
        gold_answer = item.get("gold") or item.get("gold_answer") or item.get("answer", [])
        if isinstance(gold_answer, str):
            gold_answer = [gold_answer]

        print(f"\n>>> [q{idx}]")
        print("=" * 50)
        print(f"QUERY: {query}")
        print(f"GOLD ANSWER: {gold_answer}")
        print("=" * 50)

        t_start = time.time()

        # Run pipeline 1 generation (handles hybrid retrieval, reranking, and qwen2.5)
        res = generate_answer(query)
        generated_answer = res["answer"]
        total_time = time.time() - t_start

        match = evaluate_answer(generated_answer, gold_answer)
        if match:
            total_matches += 1
            eval_status = "✅ MATCH"
        else:
            eval_status = "❌ MISMATCH"

        print(f"\n--- GENERATED ANSWER ({total_time:.2f}s) ---")
        print(generated_answer)
        print(f"\nEVALUATION: {eval_status}")

        results_summary.append({
            "id": f"q{idx}",
            "score": round(res.get("hybrid_score", 0.0), 4),
            "match": "OK" if match else "FAIL",
            "time": round(total_time, 2)
        })

    # Summary Table Output
    print("\n" + "=" * 50)
    print("            PIPELINE 1 BENCHMARK SUMMARY            ")
    print("=" * 50)
    print(f"{'ID':<8} {'Score':<10} {'Match':<8} {'Time(s)':<8}")
    print("-" * 52)
    for res in results_summary:
        print(f"{res['id']:<8} {res['score']:<10} {res['match']:<8} {res['time']:<8}")
    print("-" * 52)
    accuracy = (total_matches / len(eval_data)) * 100
    print(f"TOTAL ACCURACY: {total_matches}/{len(eval_data)} ({accuracy:.1f}%)")
    print(f"Total time elapsed: {time.time() - start_time_all:.2f}s")


if __name__ == "__main__":
    main()