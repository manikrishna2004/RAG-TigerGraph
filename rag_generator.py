import re
import sys
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag_retriever import (
    build_retrieval_verdict,
    get_dynamic_fetch_k,
    get_dynamic_k,
    hybrid_search,
)


PIPELINE_1_SYSTEM_PROMPT = """You are a precise, evidence-grounded factual assistant answering questions about Olympic sports history.

Answer ONLY from the retrieved context below.

STRICT VERIFICATION RULES:

1. ATTRIBUTE VERIFICATION
   Before using any chunk, verify the requested:
   - Gender (Men/Women)
   - Event Type (Individual/Team/Relay)
   - Distance
   - Discipline/Event
   - Year
   exactly matches the question whenever the attribute is present or implied.

2. DISCARD ATTRIBUTE COLLISIONS
   If a retrieved chunk refers to a different gender, event variant, distance,
   discipline, event type, or year, DISCARD it.
   Never substitute, extrapolate, or silently merge similar events.

3. EXHAUSTIVE AGGREGATION
   For questions containing "how many", "list all", "total", "highest",
   "most", "count", or equivalent aggregation intent:
   - inspect every provided chunk;
   - identify every verified matching event;
   - deduplicate repeated event records;
   - only then calculate the final count/comparison.

4. NO DOUBLE COUNTING
   The same event may appear in overlapping or duplicate chunks.
   Count each distinct event only once.

5. NO OUTSIDE KNOWLEDGE
   Do not fill missing records from memory or from common Olympic knowledge.
   Use only the supplied context.

6. EVIDENCE
   Cite supporting evidence with [Doc X].

7. INSUFFICIENT EVIDENCE
   If the retrieved context does not establish the exact answer,
   explicitly say that the available context is insufficient.
   Do not guess.

8. AGGREGATION TRANSPARENCY
   For count/list/highest/most questions, briefly state the verified
   matching records used for the calculation when practical.

9. TEMPORAL RESOLUTION
   The original user wording is authoritative for the answer.
   A rewritten search query is only a retrieval aid; do not let query rewriting
   change the requested facts.

Retrieved context:
{context}
"""


def format_docs(results):
    """Format chunks with stable [Doc X] identifiers."""
    blocks = []

    for i, result in enumerate(results, start=1):
        source = result["metadata"].get("source", "Unknown Source")
        chunk_id = result["metadata"].get("chunk_id", "Unknown Chunk")
        blocks.append(
            f"[Doc {i}] Source: {source} (Chunk: {chunk_id})\n"
            f"Content: {result['content']}"
        )

    return "\n\n".join(blocks)


def rewrite_query(query: str, llm) -> str:
    """
    Rewrite only temporal references needed for retrieval.
    The user's original question remains the answer-generation question.
    """
    prompt = (
        "Rewrite this question into a direct, self-contained search query. "
        "Only resolve temporal references. "
        "If it says 'the Olympics immediately before [YEAR]', resolve that "
        "to the exact previous Olympic year. "
        "Preserve every other attribute exactly, including gender, event, "
        "distance, event type, and counting intent. "
        "Return ONLY the rewritten query.\n\n"
        f"Question: {query}"
    )
    return llm.invoke(prompt).content.strip()


def generate_answer(query: str, top_k: Optional[int] = None):
    llm = ChatOllama(
        model="qwen2.5:7b-instruct",
        temperature=0,
        num_predict=900,
        num_ctx=4096,
        timeout=90,
    )

    if re.search(r"\bimmediately\s+before\s+\d{4}\b", query, re.IGNORECASE):
        search_query = rewrite_query(query, llm)
    else:
        search_query = query

    effective_top_k = (
        get_dynamic_k(search_query, default_k=5)
        if top_k is None
        else int(top_k)
    )

    effective_fetch_k = get_dynamic_fetch_k(
        search_query,
        default_k=max(30, effective_top_k * 5),
    )

    results = hybrid_search(
        search_query,
        top_k=effective_top_k,
        fetch_k=effective_fetch_k,
        enable_rerank=True,
    )

    verdict = build_retrieval_verdict(results)

    if not verdict["contains_answer"] and not results:
        return {
            "answer": (
                "I could not find sufficient information in the document "
                "database to answer your question."
            ),
            "sources": [],
            "verdict": verdict,
            "retrieved_chunks": 0,
            "search_query": search_query,
            "tokens_used": 0,
        }

    context_str = format_docs(results)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PIPELINE_1_SYSTEM_PROMPT),
            (
                "human",
                "User question:\n{question}\n\n"
                "Search query used only for retrieval:\n{search_query}",
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "context": context_str,
            "question": query,
            "search_query": search_query,
        }
    )

    meta = getattr(response, "response_metadata", {}) or {}
    eval_count = meta.get("eval_count", 0) or 0
    prompt_eval_count = meta.get("prompt_eval_count", 0) or 0
    total_tokens = eval_count + prompt_eval_count

    return {
        "answer": response.content,
        "sources": [
            result["metadata"].get("source")
            for result in results
        ],
        "hybrid_score": verdict["best_score"],
        "retrieved_chunks": len(results),
        "search_query": search_query,
        "tokens_used": total_tokens if total_tokens > 0 else "N/A",
    }


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your question: ")
    result = generate_answer(query)

    print("\n=== FINAL ANSWER ===")
    print(result["answer"])
    print("\n=== RETRIEVED SOURCES ===")
    for source in set(result.get("sources", [])):
        print(f"- {source}")
    print(f"\n=== RETRIEVED CHUNKS ===")
    print(result.get("retrieved_chunks", 0))


if __name__ == "__main__":
    main()