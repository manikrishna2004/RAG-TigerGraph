import asyncio
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, List, Dict, Literal

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from rag_retriever import hybrid_search
from graph_retriever import (graph_chunk_search, find_entities_in_question, attributes_text)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UNIFORM_MODEL_NAME = "qwen2.5:7b-instruct"
MAX_ROUNDS = 3               # 1 first pass + up to 2 targeted retries
LLM_TIMEOUT = 240.0          # CPU / cold-load safe. Warm first: ollama run <model> "hi"
NUM_CTX = 6144
VECTOR_TOP_K = 4
GRAPH_CHUNKS = 3
MAX_CHUNK_CHARS = 1800
MAX_CONTEXT_CHARS = 9000     # ~2.3k tokens of documents per LLM call
VECTOR_MIN_SCORE = 0.15      # hybrid_search default threshold 0.5 often returns nothing

EXECUTOR = ThreadPoolExecutor(max_workers=4)

_RELATIVE = re.compile(r"\b(before|after|prior to|preceding|previous|previously|following|next|earlier|later|immediately)\b", re.I)
_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


def _get_llm(num_predict: int = 400) -> ChatOllama:
    return ChatOllama(model=UNIFORM_MODEL_NAME, temperature=0, num_predict=num_predict,
                      num_ctx=NUM_CTX, keep_alive="30m", format="json")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).strip()


def _parse_json_safe(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text or "").strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _count_tokens(res) -> Dict[str, int]:
    """ChatOllama does not fill response_metadata['token_usage'] (why tokens were 0)."""
    um = getattr(res, "usage_metadata", None) or {}
    md = getattr(res, "response_metadata", None) or {}
    inp = int(um.get("input_tokens") or md.get("prompt_eval_count") or 0)
    out = int(um.get("output_tokens") or md.get("eval_count") or 0)
    total = int(um.get("total_tokens") or (inp + out))
    return {"in": inp, "out": out, "total": total}


async def _llm_json(system: str, human: str, num_predict: int):
    """One LLM call -> (parsed dict, raw text, token dict, error string)."""
    try:
        res = await asyncio.wait_for(
            _get_llm(num_predict).ainvoke([SystemMessage(content=system), HumanMessage(content=human)]),
            timeout=LLM_TIMEOUT,
        )
    except Exception as e:
        return {}, "", {"in": 0, "out": 0, "total": 0}, f"{type(e).__name__}: {e}"
    return _parse_json_safe(res.content), res.content, _count_tokens(res), ""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    resolved_question: str
    next_query: str
    strategy: str                 # graph | vector | both  (chosen by the LLM after round 0)
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
    tr.append({"step": len(tr) + 1, "op": op, "seconds": round(time.time() - t0, 2),
               "tokens": (tokens or {}).get("total", 0), **detail})
    return tr


# ---------------------------------------------------------------------------
# Node 1: resolve relative references. One small LLM call, only if the question has a
# relative word AND a year (generic; no benchmark templates).
# ---------------------------------------------------------------------------
async def resolve_node(state: AgentState):
    q = state["question"]
    out = {"resolved_question": q, "next_query": q, "strategy": "both", "tools_invoked": [],
           "trace": [], "llm_tokens_used": 0, "tokens_in": 0, "tokens_out": 0}
    if not (_RELATIVE.search(q) and _YEAR.search(q)):
        return out
    t0 = time.time()
    parsed, _, tok, err = await _llm_json(
        "Rewrite the question so any relative reference (the event/edition/year before, after, "
        "previous, next ...) is replaced by the explicit name or year, using only well-known "
        "facts. Do NOT answer it. Keep everything else unchanged. "
        'Reply JSON: {"question": "<rewritten question>"}', q, 120)
    new_q = str(parsed.get("question", "")).strip()
    if new_q and not err:
        out.update(resolved_question=new_q, next_query=new_q)
    out.update(tools_invoked=["resolve_relative"], llm_tokens_used=tok["total"],
               tokens_in=tok["in"], tokens_out=tok["out"],
               trace=_step(out, "resolve_relative_reference", t0, tok,
                           detail=f"{q!r} -> {out['resolved_question']!r}" + (f" (err {err})" if err else "")))
    print(f"  [Resolve] {q!r} -> {out['resolved_question']!r}")
    return out


# ---------------------------------------------------------------------------
# Node 2: retrieval round. Specialised tools: entity linking -> graph traversal (chunk text),
# similarity search (hybrid dense+BM25+rerank). The LLM picks the strategy for retries.
# ---------------------------------------------------------------------------
def _graph_tools(query: str, n: int):
    t0 = time.time()
    seeds = find_entities_in_question(query)
    t_link = time.time() - t0
    t1 = time.time()
    res = graph_chunk_search(query, max_chunks=n, seeds=seeds) if seeds else []
    out = [{"chunk_id": r["chunk_id"], "content": r["content"]} for r in res]
    used_seeds = sorted({s for r in res for s in r["seeds"]})
    fallback = False
    if not res and seeds:
        # chunk text unresolved: steer similarity search with the graph's entity name
        fallback = True
        for r in hybrid_search(f"{seeds[0]} {query}", top_k=VECTOR_TOP_K, min_score=VECTOR_MIN_SCORE):
            out.append({"chunk_id": r["metadata"].get("chunk_id", r["metadata"].get("source", "unknown")),
                        "content": r["content"]})
    attrs = attributes_text((used_seeds or seeds)[:4]) if seeds else ""
    if attrs:
        out.insert(0, {"chunk_id": "attrs::" + "|".join((used_seeds or seeds)[:4]),
                       "content": "Graph attributes (LLM-extracted, may be inaccurate; documents win):\n" + attrs})
    return out, {"seeds": seeds, "t_link": t_link, "t_graph": time.time() - t1, "fallback": fallback}


def _vector_tool(query: str, k: int):
    t0 = time.time()
    res = hybrid_search(query, top_k=k, min_score=VECTOR_MIN_SCORE)
    return ([{"chunk_id": r["metadata"].get("chunk_id", r["metadata"].get("source", "unknown")),
              "content": r["content"]} for r in res], time.time() - t0)


async def retrieve_node(state: AgentState):
    loop = asyncio.get_running_loop()
    q = state["next_query"] or state["resolved_question"]
    strategy = state.get("strategy", "both")
    if strategy not in ("graph", "vector", "both"):
        strategy = "both"
    used = list(state.get("queries_used", []))
    widen = 3 if q in used else 0          # same query again -> go deeper, never repeat identically
    used.append(q)
    rn = state.get("round_no", 0)
    trace = list(state.get("trace", []))
    tools = list(state.get("tools_invoked", []))
    if rn > 0 and strategy != state.get("last_strategy", "both"):
        trace.append({"step": len(trace) + 1, "op": "strategy_change", "seconds": 0, "tokens": 0,
                      "detail": f"round {rn}: {state.get('last_strategy', 'both')} -> {strategy}"})

    g_fut = loop.run_in_executor(EXECUTOR, _graph_tools, q, GRAPH_CHUNKS + widen) if strategy in ("graph", "both") else None
    v_fut = loop.run_in_executor(EXECUTOR, _vector_tool, q, VECTOR_TOP_K + widen) if strategy in ("vector", "both") else None
    g_res, g_info, v_res, t_vec = [], {}, [], 0.0
    if g_fut:
        try:
            g_res, g_info = await g_fut
        except Exception as e:
            print(f"  [Graph tool error] {e}")
    if v_fut:
        try:
            v_res, t_vec = await v_fut
        except Exception as e:
            print(f"  [Vector tool error] {e}")

    seen = set(state.get("seen_ids", []))
    seen_text = {c["content"][:200] for rnd in state.get("rounds", []) for c in rnd}
    new_chunks = []
    for origin, items in (("graph", g_res), ("vector", v_res)):
        for c in items:
            key = c["content"][:200]
            if c["chunk_id"] in seen or key in seen_text:
                continue
            seen.add(c["chunk_id"])
            seen_text.add(key)
            new_chunks.append({"chunk_id": c["chunk_id"], "origin": origin,
                               "content": normalize_text(c["content"])[:MAX_CHUNK_CHARS]})

    if g_fut:
        trace.append({"step": len(trace) + 1, "op": "entity_linking", "seconds": round(g_info.get("t_link", 0), 2),
                      "tokens": 0, "detail": g_info.get("seeds", [])[:4]})
        trace.append({"step": len(trace) + 1, "op": "graph_traversal", "seconds": round(g_info.get("t_graph", 0), 2),
                      "tokens": 0, "chunks": len(g_res),
                      "detail": "chunk text unresolved -> seed-steered vector fallback" if g_info.get("fallback") else "chunk text via graph edges"})
        tools += [f"entity_linking(r{rn})", f"graph_traversal(r{rn})"]
    if v_fut:
        trace.append({"step": len(trace) + 1, "op": "similarity_search", "seconds": round(t_vec, 2),
                      "tokens": 0, "chunks": len(v_res), "detail": "dense+BM25+RRF+rerank"})
        tools.append(f"similarity_search(r{rn})")
    print(f"  [Retrieve r{rn}] strategy={strategy} q={q!r} graph={len(g_res)} vector={len(v_res)} new={len(new_chunks)}")
    return {"rounds": state.get("rounds", []) + [new_chunks], "seen_ids": list(seen),
            "queries_used": used, "tools_invoked": tools, "trace": trace, "last_strategy": strategy}


# ---------------------------------------------------------------------------
# Node 3: answer-or-search. ONE LLM call answers, cites, judges sufficiency, and picks the
# next strategy/query if evidence is missing.
# ---------------------------------------------------------------------------
def _build_docs(rounds: List[List[Dict]]):
    """Newest round first (the retry targets the missing fact), until the char budget is hit."""
    parts, ids, used = [], [], 0
    hints = []
    for rnd in reversed(rounds):
        for c in rnd:
            if c["chunk_id"].startswith("attrs::"):      # graph attribute hints: shown, not citable
                hints.append(f"[Graph hint] {c['content']}")
                continue
            block = f"[Doc {len(ids) + 1}] ({c['origin']}) {c['content']}"
            if used + len(block) > MAX_CONTEXT_CHARS and ids:
                return "\n\n".join(parts + hints[:2]), ids
            parts.append(block)
            ids.append(c["chunk_id"])
            used += len(block)
    return "\n\n".join(parts + hints[:2]), ids


async def answer_node(state: AgentState):
    t0 = time.time()
    rn = state["round_no"]
    last_round = rn >= MAX_ROUNDS - 1
    docs, ids = _build_docs(state["rounds"])
    q, rq = state["question"], state["resolved_question"]
    qtxt = q if rq == q else f"{q}\n(Interpreted as: {rq})"

    if last_round:
        system = ("Answer the question using ONLY the documents. Documents are authoritative; graph "
                  "attributes may be wrong. Give the best-supported short answer. If the documents truly "
                  "do not contain it, answer exactly 'Not found in the documents.' "
                  'Reply JSON: {"answer": "<answer>", "sufficient": true, "citations": [<doc numbers used>]}')
    else:
        system = ("Answer the question using ONLY the documents. If they contain the answer, give a short "
                  "direct answer, set sufficient=true and list the doc numbers used in citations. If not, "
                  "set sufficient=false, answer='', say what specific fact is missing, write next_query "
                  "(a short search string naming the missing entity/event/year explicitly, different from "
                  "earlier queries) and strategy: 'graph' (entity/relationship lookup), 'vector' "
                  "(free-text similarity) or 'both'. Do not guess. "
                  'Reply JSON: {"answer": "...", "sufficient": true|false, "citations": [1], '
                  '"missing": "...", "next_query": "...", "strategy": "graph|vector|both"}')

    parsed, raw, tok, err = await _llm_json(system, f"Question: {qtxt}\n\nDocuments:\n{docs}", 400)
    base = {"llm_tokens_used": state.get("llm_tokens_used", 0) + tok["total"],
            "tokens_in": state.get("tokens_in", 0) + tok["in"],
            "tokens_out": state.get("tokens_out", 0) + tok["out"],
            "tools_invoked": state.get("tools_invoked", []) + [f"answer_or_search(r{rn})"]}

    if err:
        print(f"  [LLM error] {err}")
        return {**base, "final_answer": "", "stop_reason": "llm_error", "sufficient": True, "missing": err,
                "trace": _step(state, "answer_or_search", t0, tok, detail=f"LLM error: {err}")}

    answer = normalize_text(str(parsed.get("answer", "")).strip()) if parsed else normalize_text(raw)
    sufficient = bool(parsed.get("sufficient", bool(answer))) if parsed else bool(answer)
    cites = []
    for n in (parsed.get("citations") or []) if parsed else []:
        try:
            cites.append(ids[int(n) - 1])
        except (ValueError, IndexError, TypeError):
            pass

    if last_round or sufficient:
        reason = "evidence_sufficient" if sufficient and not last_round else \
                 ("answered_on_last_round" if answer else "max_rounds_hit")
        return {**base, "final_answer": answer or "Not found in the documents.", "citations": cites,
                "stop_reason": reason, "sufficient": True,
                "trace": _step(state, "answer_or_search", t0, tok, decision="answer", chunks=len(ids), stop_reason=reason)}

    strat = str(parsed.get("strategy", "both")).strip().lower()
    nq = str(parsed.get("next_query", "")).strip() or state["resolved_question"]
    return {**base, "sufficient": False, "missing": str(parsed.get("missing", "")), "next_query": nq,
            "strategy": strat if strat in ("graph", "vector", "both") else "both", "round_no": rn + 1,
            "trace": _step(state, "answer_or_search", t0, tok, decision="search_again",
                           missing=str(parsed.get("missing", ""))[:120], next_query=nq, next_strategy=strat)}


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
    """Importable entry point for the benchmark runner. Returns the final state, incl. 'trace',
    'tokens_in/out', 'llm_tokens_used', 'citations' (chunk_ids), 'stop_reason'."""
    app = build_agentic_graph()
    state = {"question": query, "resolved_question": query, "next_query": query, "strategy": "both",
             "last_strategy": "both", "queries_used": [], "rounds": [], "seen_ids": [], "round_no": 0, "tools_invoked": [],
             "trace": [], "sufficient": False, "missing": "", "final_answer": "", "citations": [],
             "stop_reason": "", "llm_tokens_used": 0, "tokens_in": 0, "tokens_out": 0}
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
        print(f"{t['step']:>2}. {t['op']:<26} {t['seconds']:>7.2f}s  tok={t['tokens']:<5} {extra}")
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