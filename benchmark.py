import sys
import time
import json
import asyncio
import os
import re
from typing import List, Dict, Any, Optional

try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    def estimate_tokens(text: str) -> int:
        return len(_TOKENIZER.encode(text or ""))
except ImportError:
    def estimate_tokens(text: str) -> int:
        return int(len(text or "").split() * 1.3)

# Import pipeline entrypoints
try:
    from rag_generator import generate_answer as pipeline_1_rag
    from graph_generator import answer_question as pipeline_2_graphrag
    from agent_orchestrator import run_agent as pipeline_3_agentic
except ImportError as e:
    print(f"[ERROR] Import failed: {e}", flush=True)
    sys.exit(1)

from langchain_ollama import ChatOllama

DEFAULT_JSONL_FILE = "eval_public (1).jsonl"
OUTPUT_JSONL_FILE = "benchmark_results.jsonl"
OUTPUT_JSON_FILE = "benchmark_results.json"


def load_questions_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Parse JSONL file and extract structured entries including ground truth."""
    items = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    q_text = data.get("question") or data.get("q") or data.get("query") or data.get("prompt")
                    qid = data.get("id") or data.get("qid") or f"pub-{line_no:03d}"
                    gold = data.get("answer") or data.get("gold_answer") or data.get("ground_truth") or data.get("target") or "N/A"
                    if q_text:
                        items.append({"id": qid, "question": q_text, "ground_truth": gold, "raw": data})
                except json.JSONDecodeError:
                    print(f"  [Warning] Skipping malformed JSON line {line_no}", flush=True)
        print(f"  [Loader] Successfully loaded {len(items)} questions from '{filepath}'\n", flush=True)
    except FileNotFoundError:
        print(f"  [Error] File '{filepath}' not found!", flush=True)
    return items


def judge_accuracy(question: str, ground_truth: str, generated_answer: str) -> Dict[str, Any]:
    """LLM-as-a-Judge evaluating factual accuracy and completeness (0 to 100%)."""
    if not ground_truth or ground_truth == "N/A":
        return {"score": 0, "correct": False, "reason": "No ground truth available"}

    # Exact string / numeric shortcut
    gt_norm = str(ground_truth).strip().lower()
    gen_norm = str(generated_answer).strip().lower()
    if gt_norm == gen_norm or gt_norm in gen_norm:
        return {"score": 100, "correct": True, "reason": "Exact ground truth string match"}

    evaluator = ChatOllama(model="qwen2.5:7b-instruct", temperature=0, num_predict=150, format="json")
    prompt = f"""You are an unbiased academic grader.
Compare the candidate answer against the official ground truth answer.

QUESTION: {question}
GROUND TRUTH ANSWER: {ground_truth}
CANDIDATE ANSWER: {generated_answer}

Evaluate if the candidate answer is factually correct and contains the exact requested numbers/entities.
Reply ONLY with JSON:
{{"correct": true/false, "score": 0-100, "reason": "short explanation"}}
"""
    try:
        res = evaluator.invoke(prompt)
        parsed = json.loads(re.sub(r"```(?:json)?", "", res.content).strip().rstrip("`"))
        return {
            "score": int(parsed.get("score", 0)),
            "correct": bool(parsed.get("correct", False)),
            "reason": str(parsed.get("reason", "Graded by LLM"))
        }
    except Exception:
        is_corr = gt_norm in gen_norm
        return {"score": 100 if is_corr else 0, "correct": is_corr, "reason": "Fallback regex match"}


def run_p1_rag(question: str) -> Dict[str, Any]:
    """Run Pipeline 1: Hybrid Dense + BM25 + RRF Baseline."""
    start_time = time.time()
    try:
        res = pipeline_1_rag(question)
        elapsed = time.time() - start_time
        ans = res.get("answer", "").strip()
        ctx = res.get("search_query", "") + " " + " ".join(res.get("sources", []))
        toks = estimate_tokens(question) + estimate_tokens(ctx) + estimate_tokens(ans) + 250
        return {
            "answer": ans,
            "latency_s": round(elapsed, 2),
            "retrieved_chunks": res.get("retrieved_chunks", 0),
            "sources": list(set(res.get("sources", []))),
            "tokens_used": toks,
            "status": "SUCCESS"
        }
    except Exception as e:
        return {
            "answer": f"ERROR: {e}",
            "latency_s": round(time.time() - start_time, 2),
            "retrieved_chunks": 0,
            "sources": [],
            "tokens_used": 0,
            "status": "FAILED"
        }


def run_p2_graphrag(question: str) -> Dict[str, Any]:
    """Run Pipeline 2: Pure GraphRAG Baseline."""
    start_time = time.time()
    try:
        res = pipeline_2_graphrag(question, max_facts=60, depth=2)
        elapsed = time.time() - start_time
        ans = res.get("answer", "").strip()
        ctx_str = res.get("context_str", "")
        toks = estimate_tokens(question) + estimate_tokens(ctx_str) + estimate_tokens(ans) + 200
        chunks_count = len(res.get("chunks", [])) or len(res.get("facts", []))
        return {
            "answer": ans,
            "latency_s": round(elapsed, 2),
            "retrieved_chunks": chunks_count,
            "tokens_used": toks,
            "status": "SUCCESS"
        }
    except Exception as e:
        return {
            "answer": f"ERROR: {e}",
            "latency_s": round(time.time() - start_time, 2),
            "retrieved_chunks": 0,
            "tokens_used": 0,
            "status": "FAILED"
        }


async def run_p3_agentic(question: str) -> Dict[str, Any]:
    """Run Pipeline 3: Agentic GraphRAG Multi-Turn Orchestrator."""
    start_time = time.time()
    try:
        res = await pipeline_3_agentic(question)
        elapsed = time.time() - start_time
        ans = res.get("final_answer", "").strip()
        toks = res.get("llm_tokens_used", 0)
        if toks == 0:
            toks = estimate_tokens(question) * (res.get("round_no", 0) + 1) + estimate_tokens(ans) + 800
        return {
            "answer": ans,
            "latency_s": round(elapsed, 2),
            "retrieved_chunks": len(res.get("seen_ids", [])),
            "rounds": res.get("round_no", 0) + 1,
            "tools_invoked": res.get("tools_invoked", []),
            "stop_reason": res.get("stop_reason", "completed"),
            "tokens_used": toks,
            "status": "SUCCESS"
        }
    except Exception as e:
        return {
            "answer": f"ERROR: {e}",
            "latency_s": round(time.time() - start_time, 2),
            "retrieved_chunks": 0,
            "rounds": 0,
            "tools_invoked": [],
            "stop_reason": "failed",
            "tokens_used": 0,
            "status": "FAILED"
        }


def print_single_result_table(entry: Dict[str, Any]):
    """Print accuracy matrix showing score, token efficiency, and winner per question."""
    q_id = str(entry["id"])
    gt = str(entry["ground_truth"])
    print("\n" + "=" * 125, flush=True)
    print(f"📊 LIVE BENCHMARK METRICS FOR QUESTION ID: {q_id}", flush=True)
    print(f"🎯 QUESTION: {entry['question']}", flush=True)
    print(f"🔑 GROUND TRUTH: {gt}", flush=True)
    print("=" * 125, flush=True)
    print(f"{'Pipeline':<18} | {'Latency':<8} | {'Chunks':<8} | {'Tokens':<8} | {'Accuracy':<10} | {'Verdict':<8} | {'Answer Output'}", flush=True)
    print("-" * 125, flush=True)

    # P1 Row
    p1 = entry['p1_rag']
    p1_v = "✅ PASS" if p1['eval']['correct'] else "❌ FAIL"
    print(f"{'P1 (Hybrid RAG)':<18} | {p1['latency_s']}s{'':<3} | {p1['retrieved_chunks']:<8} | {p1['tokens_used']:<8} | {p1['eval']['score']}%{'':<5} | {p1_v:<8} | {p1['answer'][:30]}...", flush=True)

    # P2 Row
    p2 = entry['p2_graphrag']
    p2_v = "✅ PASS" if p2['eval']['correct'] else "❌ FAIL"
    print(f"{'P2 (GraphRAG)':<18} | {p2['latency_s']}s{'':<3} | {p2['retrieved_chunks']:<8} | {p2['tokens_used']:<8} | {p2['eval']['score']}%{'':<5} | {p2_v:<8} | {p2['answer'][:30]}...", flush=True)

    # P3 Row
    p3 = entry['p3_agentic']
    p3_v = "✅ PASS" if p3['eval']['correct'] else "❌ FAIL"
    print(f"{'P3 (Agentic)':<18} | {p3['latency_s']}s{'':<3} | {p3['retrieved_chunks']:<8} | {p3['tokens_used']:<8} | {p3['eval']['score']}%{'':<5} | {p3_v:<8} | {p3['answer'][:30]}...", flush=True)

    print("-" * 125, flush=True)
    print(f"🏆 BEST PERFORMING PIPELINE FOR THIS QUESTION: {entry['winner']}", flush=True)
    print("=" * 125 + "\n", flush=True)


async def execute_benchmark(eval_items: List[Dict[str, Any]], max_questions: int = None) -> List[Dict[str, Any]]:
    benchmark_results = []
    if max_questions:
        eval_items = eval_items[:max_questions]

    with open(OUTPUT_JSONL_FILE, "w", encoding="utf-8") as stream_f:
        pass

    print("=" * 90, flush=True)
    print(f"🚀 STARTING AGENTIC GRAPHRAG HACKATHON BENCHMARK ({len(eval_items)} Questions)", flush=True)
    print("=" * 90, flush=True)

    for idx, item in enumerate(eval_items, start=1):
        q_id = item["id"]
        q_text = item["question"]
        gt_text = item["ground_truth"]

        print(f"\n[Processing {idx}/{len(eval_items)}] (ID: {q_id}) Question: {q_text}", flush=True)

        # 1. Run Pipelines
        p1_res = run_p1_rag(q_text)
        p2_res = run_p2_graphrag(q_text)
        p3_res = await run_p3_agentic(q_text)

        # 2. Evaluate Accuracy Against Ground Truth
        p1_res["eval"] = judge_accuracy(q_text, gt_text, p1_res["answer"])
        p2_res["eval"] = judge_accuracy(q_text, gt_text, p2_res["answer"])
        p3_res["eval"] = judge_accuracy(q_text, gt_text, p3_res["answer"])

        # 3. Determine Winner
        scores = {
            "P1 (Hybrid RAG)": (p1_res["eval"]["score"], -p1_res["tokens_used"]),
            "P2 (GraphRAG)": (p2_res["eval"]["score"], -p2_res["tokens_used"]),
            "P3 (Agentic GraphRAG)": (p3_res["eval"]["score"], -p3_res["tokens_used"])
        }
        best_pipe = max(scores, key=scores.get)

        entry = {
            "id": q_id,
            "question": q_text,
            "ground_truth": gt_text,
            "p1_rag": p1_res,
            "p2_graphrag": p2_res,
            "p3_agentic": p3_res,
            "winner": best_pipe
        }
        
        benchmark_results.append(entry)

        with open(OUTPUT_JSONL_FILE, "a", encoding="utf-8") as stream_f:
            stream_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream_f.flush()

        print_single_result_table(entry)

    return benchmark_results


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSONL_FILE
    limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None

    eval_items = load_questions_from_jsonl(filepath)
    if not eval_items:
        sys.exit(1)

    results = asyncio.run(execute_benchmark(eval_items, max_questions=limit))
    
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ BENCHMARK COMPLETE! Results saved to '{OUTPUT_JSON_FILE}'", flush=True)


if __name__ == "__main__":
    main()