import asyncio
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, List, Dict, Literal

from langgraph.graph import StateGraph, END

# Native retrievers from Pipeline 1 and Pipeline 2
from rag_retriever import hybrid_search, get_dynamic_k, get_dynamic_fetch_k
from graph_retriever import graph_chunk_search, find_entities_in_question, attributes_text
from tg_connection import get_tg_connection

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UNIFORM_MODEL_NAME = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_ROUNDS = 2               
LLM_TIMEOUT = 120.0          

EXECUTOR = ThreadPoolExecutor(max_workers=5)

_RELATIVE = re.compile(
    r"\b(before|after|prior to|preceding|previous|previously|following|next|earlier|later|immediately)\b", 
    re.I
)
_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


# ---------------------------------------------------------------------------
# Direct Thread-Safe HTTP Engine
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).strip()


def _parse_json_safe(text: str) -> dict:
    if not text:
        return {}
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _call_ollama_raw(prompt: str, system: str, num_ctx: int = 8192, num_predict: int = 512, json_mode: bool = False) -> tuple[dict, str, dict, str]:
    payload = {
        "model": UNIFORM_MODEL_NAME,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
    }
    if json_mode:
        payload["format"] = "json"

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data_bytes, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            raw_text = res_json.get("response", "").strip()

            p_tokens = res_json.get("prompt_eval_count")
            e_tokens = res_json.get("eval_count")

            if p_tokens is None or p_tokens == 0:
                p_tokens = len(system + prompt) // 4
            if e_tokens is None or e_tokens == 0:
                e_tokens = len(raw_text) // 4

            tok = {
                "in": int(p_tokens),
                "out": int(e_tokens),
                "total": int(p_tokens) + int(e_tokens)
            }

            parsed = _parse_json_safe(raw_text) if json_mode else {}
            return parsed, raw_text, tok, ""
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        return {}, "", {"in": 0, "out": 0, "total": 0}, err_msg


async def _llm_call(system: str, human: str, num_ctx: int = 8192, num_predict: int = 512, json_mode: bool = False):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        EXECUTOR, _call_ollama_raw, human, system, num_ctx, num_predict, json_mode
    )


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    resolved_question: str
    next_query: str
    strategy: str                 
    last_strategy: str
    queries_used: List[str]
    rounds: List[List[Dict]]
    seen_ids: List[str]
    round_no: int
    tools_invoked: List[str]
    trace: List[Dict]
    sufficient: bool
    missing: str
    final_answer: str
    citations: List[str]
    stop_reason: str
    llm_tokens_used: int
    tokens_in: int
    tokens_out: int


def _step(state, op, t0, tokens=None, **detail):
    tr = list(state.get("trace", []))
    tr.append({
        "step": len(tr) + 1, 
        "op": op, 
        "seconds": round(time.time() - t0, 2),
        "tokens": (tokens or {}).get("total", 0) if isinstance(tokens, dict) else (tokens or 0), 
        **detail
    })
    return tr


# ---------------------------------------------------------------------------
# Node 1: Resolve Relative References
# ---------------------------------------------------------------------------
async def resolve_node(state: AgentState):
    q = state["question"]
    out = {
        "resolved_question": q, "next_query": q, "strategy": "both", 
        "tools_invoked": [], "trace": [], "llm_tokens_used": 0, 
        "tokens_in": 0, "tokens_out": 0
    }
    if not (_RELATIVE.search(q) and _YEAR.search(q)):
        return out

    t0 = time.time()
    system_prompt = (
        "Rewrite the question so any relative reference (e.g. 'the Olympics immediately before 2016') "
        "is replaced by the explicit target year or event name using standard Olympic historical facts. "
        "Preserve all other details (gender, discipline, distance, count intent) exactly. Do NOT answer it. "
        'Reply JSON: {"question": "<rewritten question>"}'
    )
    parsed, _, tok, err = await _llm_call(system_prompt, q, num_ctx=2048, num_predict=120, json_mode=True)
    new_q = str(parsed.get("question", "")).strip() if parsed else ""
    
    if new_q and not err:
        out.update(resolved_question=new_q, next_query=new_q)

    out.update(
        tools_invoked=["resolve_relative"], 
        llm_tokens_used=tok["total"],
        tokens_in=tok["in"], 
        tokens_out=tok["out"],
        trace=_step(out, "resolve_relative_reference", t0, tok, detail=f"{q!r} -> {out['resolved_question']!r}")
    )
    return out


# ---------------------------------------------------------------------------
# Tools & Execution
# ---------------------------------------------------------------------------
def _pipeline1_tool(query: str):
    t0 = time.time()
    top_k = get_dynamic_k(query)
    fetch_k = get_dynamic_fetch_k(query)
    
    res = hybrid_search(query, top_k=top_k, fetch_k=fetch_k, enable_rerank=True, min_score=0.1)
    out = []
    for r in res:
        meta = r.get("metadata", {})
        cid = meta.get("chunk_id") or meta.get("source") or "unknown"
        out.append({
            "chunk_id": cid,
            "content": r.get("content", ""),
            "score": r.get("final_score", 0.0)
        })
    return out, time.time() - t0


def _pipeline2_tool(query: str):
    t0 = time.time()
    seeds = find_entities_in_question(query)
    t_link = time.time() - t0
    t1 = time.time()
    
    res = graph_chunk_search(query, max_chunks=5, seeds=seeds) if seeds else []
    out = [{"chunk_id": r["chunk_id"], "content": r["content"]} for r in res]
    
    used_seeds = sorted({s for r in res for s in r["seeds"]}) if res else []
    attrs = attributes_text((used_seeds or seeds)[:4]) if seeds else ""
    if attrs:
        out.insert(0, {
            "chunk_id": "attrs::" + "|".join((used_seeds or seeds)[:4]),
            "content": "Graph attributes:\n" + attrs
        })
    return out, {"seeds": seeds, "t_link": t_link, "t_graph": time.time() - t1}


def _aggregation_tool(query: str) -> str:
    """Phase 3 Tool: Directly pulls metrics from TigerGraph for counting queries."""
    triggers = {"how many", "highest", "most", "total", "count", "which"}
    q_lower = query.lower()
    
    if not any(t in q_lower for t in triggers):
        return ""
        
    years = re.findall(r"\b(?:19|20)\d{2}\b", q_lower)
    target_year = int(years[0]) if years else None
    
    q_words = set(re.findall(r'[a-z]+', q_lower))
    stop = {"how","many","which","event","events","the","in","at","of","had","more","than","highest","number","competitors","nations","according","provided","corpus"}
    q_words = q_words - stop
    
    try:
        conn = get_tg_connection()
        count = conn.getVertexCount("Entity")
        verts = conn.getVertices("Entity", limit=max(int(count), 1))
        
        records = []
        for v in (verts or []):
            attrs = v.get("attributes", {})
            comp = attrs.get("competitors", 0)
            nat = attrs.get("nations", 0)
            
            if comp > 0 or nat > 0:
                yr = attrs.get("year", 0)
                if target_year and yr != 0 and yr != target_year:
                    continue
                    
                vid = str(v.get("v_id", ""))
                v_words = set(re.findall(r'[a-z]+', vid.lower()))
                
                # Filter strictly if there's sport/event keyword overlap
                if not q_words or len(q_words & v_words) > 0:
                    records.append(f"Event: {vid} | Competitors: {comp} | Nations: {nat} | Year: {yr}")
        
        if records:
            return "DATABASE METRICS (EXACT COUNTS):\n" + "\n".join(records[:100])
    except Exception as e:
        print(f"  [Aggregation Tool Error]: {e}")
        
    return ""


# ---------------------------------------------------------------------------
# Node 2: Retrieval Round (with Phase 2 RRF Fusion)
# ---------------------------------------------------------------------------
async def retrieve_node(state: AgentState):
    loop = asyncio.get_running_loop()
    q = state["next_query"] or state["resolved_question"]
    strategy = state.get("strategy", "both")
    used = list(state.get("queries_used", []))
    used.append(q)
    
    rn = state.get("round_no", 0)
    trace = list(state.get("trace", []))
    tools = list(state.get("tools_invoked", []))

    # Concurrently execute all three specialized tools
    agg_fut = loop.run_in_executor(EXECUTOR, _aggregation_tool, q)
    p1_fut = loop.run_in_executor(EXECUTOR, _pipeline1_tool, q) if strategy in ("pipeline1", "vector", "both") else None
    p2_fut = loop.run_in_executor(EXECUTOR, _pipeline2_tool, q) if strategy in ("pipeline2", "graph", "both") else None

    agg_res = await agg_fut if agg_fut else ""
    p1_res, t_p1, p2_res, p2_info = [], 0.0, [], {}
    
    if p1_fut:
        try:
            p1_res, t_p1 = await p1_fut
        except Exception as e:
            print(f"  [Pipeline 1 Error] {e}")
    if p2_fut:
        try:
            p2_res, p2_info = await p2_fut
        except Exception as e:
            print(f"  [Pipeline 2 Error] {e}")

    # Phase 2: Reciprocal Rank Fusion (RRF) deduplication
    attrs_chunk = None
    if p2_res and p2_res[0]["chunk_id"].startswith("attrs::"):
        attrs_chunk = p2_res.pop(0)

    p1_ranks = {c["chunk_id"]: i for i, c in enumerate(p1_res)}
    p2_ranks = {c["chunk_id"]: i for i, c in enumerate(p2_res)}
    
    fused = {}
    for c in p1_res + p2_res:
        cid = c["chunk_id"]
        if cid not in fused:
            fused[cid] = {"chunk_id": cid, "content": c["content"], "origin": set()}
        origin = "pipeline1" if cid in p1_ranks else "pipeline2"
        fused[cid]["origin"].add(origin)
        
    K = 60
    scored_candidates = []
    for cid, data in fused.items():
        score = 0.0
        if cid in p1_ranks:
            score += 1.0 / (K + p1_ranks[cid] + 1)
        if cid in p2_ranks:
            score += 1.0 / (K + p2_ranks[cid] + 1)
        scored_candidates.append((score, data))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Construct final chunks for the LLM context window
    new_chunks = []
    if agg_res:
        new_chunks.append({"chunk_id": "DB_METRICS", "origin": "aggregation_tool", "content": agg_res})
        tools.append(f"aggregation_query(r{rn})")
        
    if attrs_chunk:
        new_chunks.append({"chunk_id": attrs_chunk["chunk_id"], "origin": "pipeline2_graph", "content": attrs_chunk["content"]})
        
    seen = set(state.get("seen_ids", []))
    seen_text = {c["content"][:200] for rnd in state.get("rounds", []) for c in rnd}
    
    # Grab the top 10 unique chunks after fusion
    for _, data in scored_candidates[:10]:
        key = data["content"][:200]
        if data["chunk_id"] in seen or key in seen_text:
            continue
        seen.add(data["chunk_id"])
        seen_text.add(key)
        new_chunks.append({
            "chunk_id": data["chunk_id"],
            "origin": " + ".join(data["origin"]),
            "content": normalize_text(data["content"])
        })

    if p1_fut:
        trace.append({"step": len(trace) + 1, "op": "pipeline1_hybrid_search", "seconds": round(t_p1, 2), "chunks": len(p1_res)})
        tools.append(f"pipeline1_search(r{rn})")

    if p2_fut:
        trace.append({"step": len(trace) + 1, "op": "pipeline2_graph_traversal", "seconds": round(p2_info.get("t_graph", 0), 2), "chunks": len(p2_res)})
        tools.append(f"pipeline2_graph(r{rn})")

    return {
        "rounds": state.get("rounds", []) + [new_chunks], 
        "seen_ids": list(seen),
        "queries_used": used, 
        "tools_invoked": tools, 
        "trace": trace, 
        "last_strategy": strategy
    }


# ---------------------------------------------------------------------------
# Node 3: System Prompts & Early-Stopping Answer Evaluator
# ---------------------------------------------------------------------------
EVAL_SYSTEM_PROMPT = """You are an evidence evaluator for Olympic sports history.
Inspect the retrieved context and determine if it contains evidence to answer the question.

Rules:
1. Match Gender, Event Type, Distance, Discipline, and Year exactly.
2. If evidence is complete, set sufficient = true.
3. If evidence is incomplete, set sufficient = false and propose a specific targeted search query.

PREVIOUS QUERIES USED: {queries_used}
CRITICAL: You MUST NOT recommend any query from the list above. If the previous queries failed to return the needed facts, you must change your search strategy.

Reply ONLY in valid JSON:
{{
  "sufficient": true | false,
  "missing_fact": "missing piece of evidence",
  "recommended_strategy": "pipeline1" | "pipeline2" | "both",
  "targeted_next_query": "short query to find missing fact"
}}"""


FINAL_ANSWER_SYSTEM_PROMPT = """You are a precise, evidence-grounded factual assistant answering questions about Olympic sports history.

Answer ONLY from the retrieved context below.

STRICT RULES:
1. Verify Gender, Discipline, Event Type, Distance, and Year against the question.
2. Extract facts matching all attributes. Discard non-matching details.
3. CRITICAL: If you see a [DATABASE METRICS (EXACT COUNTS)] block, you MUST use those exact metrics to answer questions like "How many events" or "Which event had the highest...". Do not manually count individual text chunks if database metrics are provided.
4. Support key statements with [Doc X] citations.
5. If context is insufficient, state 'Not found in the documents.' Do not guess.

Retrieved Context:
{context}
"""


def _build_unified_context(rounds: List[List[Dict]]) -> tuple[str, List[str]]:
    parts, doc_ids = [], []
    doc_idx = 1
    
    for rnd in rounds:
        for c in rnd:
            chunk_id = c["chunk_id"]
            if chunk_id == "DB_METRICS" or chunk_id.startswith("attrs::"):
                parts.append(f"[System Entity Data]:\n{c['content']}")
                continue
                
            origin = c.get("origin", "retrieval")
            block = f"[Doc {doc_idx}] ({origin} | Chunk: {chunk_id})\n{c['content']}"
            parts.append(block)
            doc_ids.append(chunk_id)
            doc_idx += 1
            
    return "\n\n".join(parts), doc_ids


async def answer_node(state: AgentState):
    t0 = time.time()
    rn = state["round_no"]
    last_round = rn >= MAX_ROUNDS - 1
    
    rounds = state.get("rounds", [])
    latest_chunks = rounds[-1] if rounds else []
    
    q_used = state.get("queries_used", [])
    current_q = state.get("next_query", "")
    is_duplicate_query = (rn > 0 and q_used.count(current_q) > 1)
    
    context_str, doc_ids = _build_unified_context(rounds)
    q = state["question"]
    rq = state["resolved_question"]
    qtxt = q if rq == q else f"{q}\n(Interpreted as: {rq})"

    force_final = last_round or is_duplicate_query

    if not force_final:
        search_feedback = ""
        if rn > 0 and len(latest_chunks) == 0:
            search_feedback = f"\n[SYSTEM ALERT]: The last search query '{current_q}' returned ZERO new documents. You MUST try a completely different search strategy.\n"
        
        eval_prompt = f"User Question:\n{qtxt}\n{search_feedback}\nRetrieved Context:\n{context_str}"
        formatted_eval_system = EVAL_SYSTEM_PROMPT.replace("{queries_used}", json.dumps(q_used))

        parsed_eval, _, tok_eval, err = await _llm_call(
            formatted_eval_system, eval_prompt, num_ctx=4096, num_predict=120, json_mode=True
        )
        
        base_eval = {
            "llm_tokens_used": state.get("llm_tokens_used", 0) + tok_eval["total"],
            "tokens_in": state.get("tokens_in", 0) + tok_eval["in"],
            "tokens_out": state.get("tokens_out", 0) + tok_eval["out"],
        }

        is_sufficient = parsed_eval.get("sufficient", False) if not err else False
        next_q = str(parsed_eval.get("targeted_next_query", rq)).strip()
        
        if next_q in q_used or not next_q:
            is_sufficient = True

        if not is_sufficient:
            missing = str(parsed_eval.get("missing_fact", "additional details required"))
            strat = str(parsed_eval.get("recommended_strategy", "both"))
            
            return {
                **base_eval,
                "sufficient": False,
                "missing": missing,
                "next_query": next_q if next_q else rq,
                "strategy": strat if strat in ("pipeline1", "pipeline2", "both") else "both",
                "round_no": rn + 1,
                "tools_invoked": state.get("tools_invoked", []) + [f"eval_check(r{rn})"],
                "trace": _step(
                    state, "evaluate_sufficiency", t0, tok_eval, 
                    decision="search_retry", missing=missing, next_query=next_q, strategy=strat
                )
            }

    current_num_ctx = 8192 if rn == 0 else 12288
    formatted_system = FINAL_ANSWER_SYSTEM_PROMPT.format(context=context_str if context_str else "(No documents retrieved)")
    human_prompt = f"User Question:\n{qtxt}"
    
    _, raw_answer, tok_ans, err = await _llm_call(
        formatted_system, human_prompt, num_ctx=current_num_ctx, num_predict=600, json_mode=False
    )
    
    stop_reason = "evidence_sufficient"
    if is_duplicate_query:
        stop_reason = "duplicate_query_stopped"
    elif last_round:
        stop_reason = "max_rounds_hit"

    cited_nums = re.findall(r"\[Doc\s+(\d+)\]", raw_answer)
    citations = []
    for num in cited_nums:
        try:
            idx = int(num) - 1
            if 0 <= idx < len(doc_ids):
                citations.append(doc_ids[idx])
        except ValueError:
            pass

    return {
        "llm_tokens_used": state.get("llm_tokens_used", 0) + tok_ans["total"],
        "tokens_in": state.get("tokens_in", 0) + tok_ans["in"],
        "tokens_out": state.get("tokens_out", 0) + tok_ans["out"],
        "tools_invoked": state.get("tools_invoked", []) + [f"generate_final_answer(r{rn})"],
        "final_answer": raw_answer.strip() if raw_answer else "Not found in the documents.",
        "citations": list(dict.fromkeys(citations)),
        "sufficient": True,
        "stop_reason": stop_reason,
        "trace": _step(state, "generate_final_answer", t0, tok_ans, decision="answer_complete", docs_used=len(doc_ids))
    }


def route_after_answer(state: AgentState) -> Literal["retrieve", "__end__"]:
    return END if state.get("sufficient") else "retrieve"


def build_agentic_graph():
    wf = StateGraph(AgentState)
    wf.add_node("resolve", resolve_node)
    wf.add_node("retrieve", retrieve_node)
    wf.add_node("answer", answer_node)
    wf.set_entry_point("resolve")
    wf.add_edge("resolve", "retrieve")
    wf.add_edge("retrieve", "answer")
    wf.add_conditional_edges("answer", route_after_answer, {"retrieve": "retrieve", END: END})
    return wf.compile()


async def run_agent(query: str) -> dict:
    app = build_agentic_graph()
    state = {
        "question": query, "resolved_question": query, "next_query": query, 
        "strategy": "both", "last_strategy": "both", "queries_used": [], 
        "rounds": [], "seen_ids": [], "round_no": 0, "tools_invoked": [],
        "trace": [], "sufficient": False, "missing": "", "final_answer": "", 
        "citations": [], "stop_reason": "", "llm_tokens_used": 0, 
        "tokens_in": 0, "tokens_out": 0
    }
    return await app.ainvoke(state)


async def run_main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter question: ")
    start = time.time()
    result = await run_agent(query)
    elapsed = time.time() - start
    print("\n================ FINAL ANSWER ================")
    print(result["final_answer"])
    print("\n================ AGENT TRACE ================")
    for t in result.get("trace", []):
        extra = {k: v for k, v in t.items() if k not in ("step", "op", "seconds", "tokens")}
        tokens_val = t.get("tokens", 0)
        print(f"{t['step']:>2}. {t['op']:<26} {t['seconds']:>7.2f}s  tok={tokens_val:<5} {extra}")

    print("\n================ BENCHMARK METRICS ================")
    print(f"Interpreted question:     {result.get('resolved_question')}")
    print(f"Queries used:             {result.get('queries_used')}")
    print(f"Tools Invoked (in order): {result.get('tools_invoked')}")
    print(f"Rounds used:              {result.get('round_no', 0) + 1}")
    print(f"Citations (chunk_ids):    {result.get('citations')}")
    print(f"Stop Reason:              {result.get('stop_reason')}")
    print(f"Tokens in / out / total:  {result.get('tokens_in', 0)} / {result.get('tokens_out', 0)} / {result.get('llm_tokens_used', 0)}")
    print(f"Latency Time Taken:       {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(run_main())