import argparse
import hashlib
import json
import os
import pickle
import re
import time

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from tg_connection import get_tg_connection


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Single source of truth for vector/graph chunking.
# graph_builder.py should import split_documents + these constants so the
# Chunk vertices and graph chunks retain identical chunk_ids.
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000  # Optimized from 2500 for higher retrieval precision
CHUNK_OVERLAP = 150

# Semantic/record-aware chunking settings.
ATOMIC_RECORD_MAX_CHARS = 3000
BM25_INDEX_FILENAME = "bm25_index.pkl"
BM25_INDEX_PATH = os.path.join(BASE_DIR, BM25_INDEX_FILENAME)


# ---------------------------------------------------------------------------
# Loading (Fix A: Real PDF and DOCX Loaders)
# ---------------------------------------------------------------------------
def load_documents(docs_path="docs"):
    if not os.path.isabs(docs_path):
        docs_path = os.path.join(BASE_DIR, docs_path)

    print(f"Loading documents from {docs_path}...")

    if not os.path.isdir(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    supported_extensions = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".jsonl",
        ".html",
        ".htm",
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
    }

    documents = []

    for root, _, filenames in os.walk(docs_path):
        for filename in sorted(filenames):
            filepath = os.path.join(root, filename)
            extension = os.path.splitext(filename)[1].lower()

            if extension not in supported_extensions:
                continue

            try:
                if extension == ".jsonl":
                    with open(filepath, "r", encoding="utf-8") as file:
                        for line_number, line in enumerate(file, start=1):
                            if not line.strip():
                                continue

                            obj = json.loads(line)
                            extra = (
                                {
                                    "title": str(obj.get("title", "")),
                                    "doc_id": str(obj.get("doc_id", "")),
                                }
                                if isinstance(obj, dict)
                                else {}
                            )

                            documents.append(
                                Document(
                                    page_content=json.dumps(
                                        obj,
                                        ensure_ascii=False,
                                    ),
                                    metadata={
                                        "source": filepath,
                                        "file_type": extension,
                                        "line": line_number,
                                        **extra,
                                    },
                                )
                            )
                    continue

                if extension == ".json":
                    with open(filepath, "r", encoding="utf-8") as file:
                        content = json.dumps(
                            json.load(file),
                            ensure_ascii=False,
                            indent=2,
                        )

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": filepath,
                                "file_type": extension,
                            },
                        )
                    )
                    continue

                if extension in {".txt", ".md", ".html", ".htm"}:
                    loaded = TextLoader(
                        filepath,
                        encoding="utf-8",
                    ).load()

                    for document in loaded:
                        document.metadata.update(
                            {
                                "source": filepath,
                                "file_type": extension,
                            }
                        )

                    documents.extend(loaded)
                    continue

                # Fix A Implementation: Active Loaders for PDFs and Word Files
                if extension == ".pdf":
                    loaded = PyPDFLoader(filepath).load()
                    for document in loaded:
                        document.metadata.update(
                            {
                                "source": filepath,
                                "file_type": extension,
                            }
                        )
                    documents.extend(loaded)
                    continue

                if extension == ".docx":
                    loaded = Docx2txtLoader(filepath).load()
                    for document in loaded:
                        document.metadata.update(
                            {
                                "source": filepath,
                                "file_type": extension,
                            }
                        )
                    documents.extend(loaded)
                    continue

                if extension in {".pptx", ".xlsx"}:
                    print(
                        f"  [Warning] {filename}: specialized loader not configured; skipped."
                    )

            except Exception as error:
                print(f"Skipped {filepath}: {error}")

    print(f"Total documents loaded: {len(documents)}")
    return documents


# ---------------------------------------------------------------------------
# Semantic / atomic chunking
# ---------------------------------------------------------------------------
def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_atomic_blocks(text: str):
    """
    Treat double-newline-separated records as atomic units.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return []

    return [
        block.strip()
        for block in re.split(r"\n\s*\n+", normalized)
        if block.strip()
    ]


def _fallback_split_large_block(
    block: str,
    metadata,
    chunk_size: int,
    chunk_overlap: int,
):
    """Only split a semantic record when that record itself is too large."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n", " ", ""],
    )
    docs = splitter.create_documents([block], metadatas=[metadata.copy()])
    return docs


def split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
):
    print(
        "Splitting documents using semantic blank-line boundaries "
        f"(fallback size={chunk_size}, overlap={chunk_overlap})..."
    )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    chunks = []

    for doc in documents:
        base_metadata = doc.metadata.copy()
        text = _normalize_text(doc.page_content)

        if not text:
            continue

        try:
            md_splits = markdown_splitter.split_text(text)
        except Exception:
            md_splits = [Document(page_content=text, metadata={})]

        for split in md_splits:
            header_context = " > ".join(
                str(value)
                for value in split.metadata.values()
                if value
            )

            section_text = split.page_content.strip()
            if header_context:
                section_text = (
                    f"Context: {header_context}\n\n{section_text}"
                )

            semantic_blocks = _split_atomic_blocks(section_text)

            if not semantic_blocks:
                continue

            for block in semantic_blocks:
                if len(block) <= ATOMIC_RECORD_MAX_CHARS:
                    chunks.append(
                        Document(
                            page_content=block,
                            metadata=base_metadata.copy(),
                        )
                    )
                else:
                    chunks.extend(
                        _fallback_split_large_block(
                            block=block,
                            metadata=base_metadata,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                    )

    # Fix C Implementation: Stable, content-derived ID scheme completely independent of loop index `i`.
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        content_snippet = chunk.page_content.strip()
        digest = hashlib.sha1(
            f"{source}:{content_snippet}".encode("utf-8")
        ).hexdigest()[:12]

        chunk.metadata["chunk_id"] = (
            f"{os.path.basename(source)}::{digest}"
        )

    print(f"Total chunks created: {len(chunks)}")
    print(
        f"Atomic record target <= {ATOMIC_RECORD_MAX_CHARS} chars; "
        f"fixed-size splitting is only the overflow fallback."
    )
    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def get_embedding_model():
    return OllamaEmbeddings(model="nomic-embed-text")


# ---------------------------------------------------------------------------
# Persistent corpus-wide BM25 index
# ---------------------------------------------------------------------------
def build_bm25_index(chunks):
    """Create the independent lexical index used by Pipeline 1."""
    tokenized_corpus = [
        re.findall(r"[a-z0-9]+", chunk.page_content.lower())
        for chunk in chunks
    ]

    metadata = [
        {
            "chunk_id": chunk.metadata.get("chunk_id"),
            "source": chunk.metadata.get("source", "unknown"),
            "content": chunk.page_content,
        }
        for chunk in chunks
    ]

    BM25Okapi(tokenized_corpus)

    payload = {
        "version": 1,
        "tokenized_corpus": tokenized_corpus,
        "metadata": metadata,
        "created_at": time.time(),
    }

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"✅ BM25 index written to {BM25_INDEX_PATH} "
        f"({len(metadata)} chunks)"
    )


# ---------------------------------------------------------------------------
# TigerGraph vector store
# ---------------------------------------------------------------------------
def create_tigergraph_vector_store(chunks, batch_size=50):
    """Embed and store chunks directly in TigerGraph."""
    conn = get_tg_connection()
    embedding_model = get_embedding_model()

    total = len(chunks)
    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch = chunks[
            batch_start:batch_start + batch_size
        ]

        batch_num = batch_start // batch_size + 1
        total_batches = (
            total + batch_size - 1
        ) // batch_size

        elapsed = time.time() - start_time
        print(
            f"Embedding & Upserting batch {batch_num}/{total_batches} "
            f"({batch_start}/{total} chunks, {elapsed:.0f}s elapsed)..."
        )

        # Prepend nomic instruction prefix for high-accuracy document embedding
        texts = [f"search_document: {chunk.page_content}" for chunk in batch]
        embeddings = embedding_model.embed_documents(texts)

        vertices = []

        for chunk, embedding in zip(batch, embeddings):
            chunk_id = chunk.metadata.get("chunk_id")
            source = chunk.metadata.get("source", "unknown")

            vertices.append(
                (
                    chunk_id,
                    {
                        "content": chunk.page_content,
                        "source": source,
                        "emb": embedding,
                    },
                )
            )

        try:
            conn.upsertVertices("Chunk", vertices)
        except Exception as error:
            print(
                f"Failed to upsert batch {batch_num} to TigerGraph: {error}"
            )

    print("\n✅ Vector store created directly in TigerGraph.")
    print(
        f"Total chunks embedded: {total} "
        f"in {time.time() - start_time:.0f}s"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RAG Document Ingestion Pipeline"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Embed only the first N chunks for a smoke test.",
    )
    args = parser.parse_args()

    mode = (
        f"TEST (limit={args.limit})"
        if args.limit
        else "FULL CORPUS"
    )
    print(f"=== RAG Document Ingestion Pipeline [{mode}] ===\n")

    docs_path = os.path.join(BASE_DIR, "docs")
    documents = load_documents(docs_path)

    chunks = split_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if args.limit:
        chunks = chunks[:args.limit]
        print(f"\n[TEST MODE] Truncated to {len(chunks)} chunks.")

    print(f"\nProcessing {len(chunks)} chunks...")

    # IMPORTANT: build both stores from exactly the same chunk objects.
    build_bm25_index(chunks)
    create_tigergraph_vector_store(chunks, batch_size=50)

    print("\n✅ Ingestion complete.")
    print(
        "Re-run the corresponding graph build too, because semantic chunking "
        "changes chunk boundaries and therefore chunk_ids."
    )


if __name__ == "__main__":
    main()