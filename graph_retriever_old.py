import argparse
import re
from tg_connection import get_tg_connection
print(get_tg_connection().getSchema()["EdgeTypes"])

# --- TigerGraph-backed graph retrieval -------------------------------------------------------
# Reads the graph graph_builder.py writes: vertex type "Entity" (id = entity name, no attributes),
# edge type "Relation" with attributes relation / source_file / chunk_id.
# Same public functions and return shape as the old .gml version, so agent_orchestrator.py is unchanged.

VERTEX_TYPE = "Entity"
EDGE_LIMIT_PER_NODE = 500      # bounds hub nodes (one REST call per node)
MAX_SEED_CANDIDATES = 30       # entity-name matches we fetch edges for, before keeping the best max_entities
MAX_EXPAND_NODES = 25          # neighbours expanded on hop 2 (only reached if hop 1 gave < max_facts)

_CONN = None
_NODE_NAMES = None             # all Entity ids, loaded once
_EDGE_CACHE = {}               # node -> list of edge records

# Words that are capitalized in questions but are never entities (fallback matcher only).
_STOPWORDS = {
    "what", "who", "whom", "whose", "which", "when", "where", "why", "how", "did", "does", "do",
    "is", "are", "was", "were", "the", "and", "for", "with", "from", "that", "this", "these",
    "those", "tell", "list", "name", "give", "show", "explain", "describe", "compare", "won",
    "win", "has", "have", "had", "can", "could", "would", "should", "about", "between",
}


def _conn():
    global _CONN
    if _CONN is None:
        _CONN = get_tg_connection()
    return _CONN


def _get_node_names():
    """All Entity ids, loaded once per process (needed for question -> entity matching)."""
    global _NODE_NAMES
    if _NODE_NAMES is None:
        conn = _conn()
        count = conn.getVertexCount(VERTEX_TYPE)
        verts = conn.getVertices(VERTEX_TYPE, limit=max(int(count), 1))
        _NODE_NAMES = [v["v_id"] for v in verts]
        print(f"  [Graph] Loaded {len(_NODE_NAMES)} entity names from TigerGraph")
    return _NODE_NAMES


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _edge_record(u, v, data):
    relation = data.get("relation", "connected_to")
    return {
        "source": u,
        "target": v,
        "relation": relation,
        "source_file": data.get("source_file", "unknown"),
        "chunk_id": data.get("chunk_id", "unknown"),
        "fact_str": f"{u} --[{relation}]--> {v}",
    }


def _edges_of(node):
    """All edges touching `node` (cached). Queries every edge type on the vertex so a
    REVERSE_EDGE, if the schema has one, brings incoming edges too; reverse edges are
    flipped back to their original direction."""
    if node in _EDGE_CACHE:
        return _EDGE_CACHE[node]
    records = []
    try:
        edges = _conn().getEdges(VERTEX_TYPE, node, limit=EDGE_LIMIT_PER_NODE)
    except Exception as e:
        print(f"  [Graph] Edge fetch failed for '{node}': {e}")
        edges = []
    for e in edges:
        etype = e.get("e_type", "")
        if "relation" not in etype.lower():
            continue
        u, v = e.get("from_id"), e.get("to_id")
        if etype.lower().startswith("reverse"):
            u, v = v, u
        records.append(_edge_record(u, v, e.get("attributes", {})))
    _EDGE_CACHE[node] = records
    return records


def find_entities_in_question(question: str, max_entities: int = 8):
    """Extract and rank graph entities from a natural language question.
    
    1. Exact Substring Match: Prefers longer, specific entity names over generic tokens.
    2. Specificity Scoring: Sorts candidates by word count & character length, then degree count.
    """
    names = _get_node_names()
    q_norm = f" {_norm(question)} "

    hits = []
    for node in names:
        n = _norm(node)
        if len(n) >= 3 and n not in _STOPWORDS and f" {n} " in q_norm:
            hits.append(node)

    # Remove candidate hits that are completely contained inside a longer candidate hit
    # e.g., keep "2012 Summer Olympics", drop "Olympics"
    hits = [
        h for h in hits
        if not any(h != o and f" {_norm(h)} " in f" {_norm(o)} " for o in hits)
    ]

# Fuzzy ratio fallback if exact substring matches yield fewer than 2 candidates
    if len(hits) < 2:
        from difflib import SequenceMatcher
        q_tokens = [t for t in _norm(question).split() if t not in _STOPWORDS and len(t) > 3]
        for node in names:
            n_norm = _norm(node)
            ratio = SequenceMatcher(None, n_norm, _norm(question)).ratio()
            if ratio > 0.65 or any(SequenceMatcher(None, n_norm, t).ratio() > 0.85 for t in q_tokens):
                if node not in hits:
                    hits.append(node)

    # Fallback to token matching if no exact long phrases match
    if not hits:
        tokens = {t.lower() for t in re.findall(r"\b[A-Z][A-Za-z0-9'\-]{2,}\b", question)} - _STOPWORDS
        if tokens:
            for node in names:
                node_toks = set(_norm(node).split())
                if tokens & node_toks:
                    hits.append(node)

    hits = hits[:MAX_SEED_CANDIDATES]

    # Rank by specificity (number of words & total length) first, then by edge connectivity
    def _specificity_score(entity_name: str) -> float:
        norm_name = _norm(entity_name)
        word_count = len(norm_name.split())
        char_len = len(norm_name)
        edge_count = len(_edges_of(entity_name))
        
        # Balance specificity (longer phrases) against graph degree
        return (word_count * 10.0) + (char_len * 0.5) + min(edge_count, 50)

    hits.sort(key=_specificity_score, reverse=True)
    return hits[:max_entities]


def _expand(seeds, max_depth: int, max_facts: int):
    """BFS out from the seeds, undirected. Edges touching a seed come first (direct)."""
    seed_set = set(seeds)
    seen, direct, indirect = set(), [], []
    visited = set(seeds)
    frontier = list(seeds)

    for depth in range(max_depth):
        if depth > 0 and len(direct) >= max_facts:
            break  # hop-2 facts would be cut off by max_facts anyway; skip the extra REST calls
        next_frontier = []
        for node in frontier:
            for rec in _edges_of(node):
                key = (rec["source"], rec["target"], rec["relation"])
                if key not in seen:
                    seen.add(key)
                    touches_seed = rec["source"] in seed_set or rec["target"] in seed_set
                    (direct if touches_seed else indirect).append(rec)
                other = rec["target"] if rec["source"] == node else rec["source"]
                if other not in visited:
                    visited.add(other)
                    next_frontier.append(other)
        frontier = next_frontier[:MAX_EXPAND_NODES]
        if not frontier:
            break

    return (direct + indirect)[:max_facts]


def search_question_subgraph(question: str, max_depth: int = 2, graph_path: str = None, max_facts: int = 60):
    """Question-level graph retrieval: find entities mentioned in a full question, then return
    their n-hop neighbourhood from TigerGraph. This is what the orchestrator calls.

    `graph_path` is accepted only so agent_orchestrator.py's call keeps working; it is ignored
    (there is no .gml file any more).
    """
    try:
        seeds = find_entities_in_question(question)
    except Exception as e:
        print(f"  [Graph] TigerGraph lookup failed: {e}")
        return []

    if not seeds:
        print(f"  [Graph] No entities from the question found in graph: '{question}'")
        return []
    print(f"  [Graph] Matched entities: {seeds}")

    return _expand(seeds, max_depth, max_facts)


def search_entity_subgraph(query_entity: str, max_depth: int = 2, graph_path: str = None, max_facts: int = 60):
    """n-hop traversal around an entity NAME (not a full question)."""
    try:
        names = _get_node_names()
    except Exception as e:
        print(f"  [Graph] TigerGraph lookup failed: {e}")
        return []

    query_norm = query_entity.strip().title()
    matching = [n for n in names if query_norm in n or n in query_norm][:MAX_SEED_CANDIDATES]
    if not matching:
        print(f"No direct entity match found for: '{query_entity}'")
        return []

    return _expand(matching, max_depth, max_facts)


def search_entities_subgraph(entities: list, max_depth: int = 2, max_facts: int = 60):
    """Graph traversal seeded by an explicit list of entity names.

    Called by the orchestrator when vector retrieval surfaces entities not present
    in the original question — enabling the multi-hop pattern:
        vector_search → extract entities → graph_traversal → answer.

    Each name is fuzzy-matched against known graph entities (case-insensitive
    substring), up to 3 matches per input name, then deduplicated before BFS.
    Falls back gracefully if the entity list is empty or no matches are found.
    """
    if not entities:
        return []

    try:
        names = _get_node_names()
    except Exception as e:
        print(f"  [Graph] TigerGraph lookup failed: {e}")
        return []

    seeds = []
    for entity in entities:
        if not entity:
            continue
        # Sanitize single quotes safely for query processing
        sanitized_entity = str(entity).replace("'", "\\'")
        norm = sanitized_entity.strip().lower()
        matching = [n for n in names
                    if norm in n.lower() or n.lower() in norm]
        seeds.extend(matching[:3])

    # Deduplicate while preserving order, then cap
    seen_s, unique_seeds = set(), []
    for s in seeds:
        if s not in seen_s:
            seen_s.add(s)
            unique_seeds.append(s)
    unique_seeds = unique_seeds[:MAX_SEED_CANDIDATES]

    if not unique_seeds:
        print(f"  [Graph] No graph entities matched from seed list: {entities}")
        return []

    print(f"  [Graph] Entity-seeded traversal from: {unique_seeds}")
    return _expand(unique_seeds, max_depth, max_facts)


def main():
    parser = argparse.ArgumentParser(description="Query the knowledge graph (TigerGraph)")
    parser.add_argument("entity", nargs="*", help="The entity (or full question) to search for")
    parser.add_argument("--question", action="store_true",
                        help="Treat the input as a full natural-language question (same path the orchestrator uses).")
    args = parser.parse_args()

    entity = " ".join(args.entity) if args.entity else input("Enter an entity to search in Graph: ")

    results = search_question_subgraph(entity) if args.question else search_entity_subgraph(entity)
    print(f"\nFound {len(results)} relationships for '{entity}':")
    for r in results:
        print(f" - {r['fact_str']}  [chunk: {r['chunk_id']}]")


if __name__ == "__main__":
    main()