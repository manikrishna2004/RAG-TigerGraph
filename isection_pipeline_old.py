import argparse
import hashlib
import json
import os
import time

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from tg_connection import get_tg_connection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Single source of truth for chunking. graph_builder.py imports these too, so the vector store
# and the knowledge graph produce IDENTICAL chunk_ids for the same text. Change them in one
# place only, and re-run BOTH builds afterwards.
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 200


def load_documents(docs_path="docs"):
    if not os.path.isabs(docs_path):
        docs_path = os.path.join(BASE_DIR, docs_path)
    print(f"Loading documents from {docs_path}...")

    if not os.path.isdir(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    supported_extensions = {
        ".txt", ".md", ".csv", ".json", ".jsonl", ".html", ".htm",
        ".pdf", ".docx", ".pptx", ".xlsx"
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
                            if line.strip():
                                obj = json.loads(line)
                                # title/doc_id ride along in metadata so continuation chunks (which lose the
                                # JSON header) still know which article they belong to. chunk_ids are unaffected.
                                extra = {"title": str(obj.get("title", "")), "doc_id": str(obj.get("doc_id", ""))} \
                                    if isinstance(obj, dict) else {}
                                documents.append(Document(
                                    page_content=json.dumps(obj, ensure_ascii=False),
                                    metadata={"source": filepath, "file_type": extension, "line": line_number, **extra},
                                ))
                    continue

                if extension == ".json":
                    with open(filepath, "r", encoding="utf-8") as file:
                        content = json.dumps(json.load(file), ensure_ascii=False, indent=2)
                    documents.append(Document(page_content=content, metadata={"source": filepath, "file_type": extension}))
                    continue

                if extension in {".txt", ".md", ".html", ".htm"}:
                    loaded = TextLoader(filepath, encoding="utf-8").load()
                    for document in loaded:
                        document.metadata.update({"source": filepath, "file_type": extension})
                    documents.extend(loaded)

                # NOTE: .pdf/.docx/.pptx/.xlsx are declared as supported above but have
                # no loader branch here — they are silently skipped. If your docs/ folder
                # contains any of these, tell me and I'll wire in the right loaders
                # (pypdf / python-docx / python-pptx / openpyxl) before we move on.

            except Exception as error:
                print(f"Skipped {filepath}: {error}")

    print(f"Total documents loaded: {len(documents)}")
    return documents


from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import os
import hashlib

def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    print(f"Splitting documents into {chunk_size}-character chunks (overlap={chunk_overlap})...")
    
    # 1. (Optional optimization) Prepend markdown headers if available to retain context
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for doc in documents:
        # Try markdown header splitting first, fallback to regular text if it fails or isn't markdown
        try:
            md_splits = markdown_splitter.split_text(doc.page_content)
            for split in md_splits:
                header_context = " > ".join([v for v in split.metadata.values() if v])
                if header_context:
                    split.page_content = f"Context: {header_context}\n\n{split.page_content}"
                split.metadata = doc.metadata.copy() # preserve original metadata like source
            chunks.extend(text_splitter.split_documents(md_splits))
        except Exception:
            chunks.extend(text_splitter.split_documents([doc]))

    # Assign stable, content-derived chunk_id (Keeps your exact hashing system intact)
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        digest = hashlib.sha1(f"{source}:{i}:{chunk.page_content[:100]}".encode("utf-8")).hexdigest()[:12]
        chunk.metadata["chunk_id"] = f"{os.path.basename(source)}::{digest}"

    print(f"Total chunks created: {len(chunks)}")
    return chunks


def get_embedding_model():
    return OllamaEmbeddings(model="nomic-embed-text")


def create_tigergraph_vector_store(chunks, batch_size=50):
    """Embed and store ALL chunks directly into TigerGraph, bypassing Chroma."""
    conn = get_tg_connection()
    embedding_model = get_embedding_model()

    total = len(chunks)
    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        elapsed = time.time() - start_time
        print(f"Embedding & Upserting batch {batch_num}/{total_batches} "
              f"({batch_start}/{total} chunks, {elapsed:.0f}s elapsed)...")

        # 1. Generate embeddings for the batch
        texts = [chunk.page_content for chunk in batch]
        embeddings = embedding_model.embed_documents(texts)

        # 2. Format the payload for pyTigerGraph: [(vertex_id, {attributes_dict}), ...]
        vertices = []
        for chunk, emb in zip(batch, embeddings):
            chunk_id = chunk.metadata.get("chunk_id")
            source = chunk.metadata.get("source", "unknown")
            
            vertices.append((
                chunk_id,
                {
                    "content": chunk.page_content,
                    "source": source,
                    "emb": emb
                }
            ))

        # 3. Upsert to TigerGraph
        try:
            conn.upsertVertices("Chunk", vertices)
        except Exception as e:
            print(f"Failed to upsert batch {batch_num} to TigerGraph: {e}")

    print(f"\n✅ Vector store created and saved directly to TigerGraph")
    print(f"Total chunks embedded: {total} in {time.time() - start_time:.0f}s")

def main():
    parser = argparse.ArgumentParser(description="RAG Document Ingestion Pipeline")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only embed the first N chunks (for a fast smoke test). Omit for full corpus.")
    args = parser.parse_args()

    mode = f"TEST (limit={args.limit})" if args.limit else "FULL CORPUS"
    print(f"=== RAG Document Ingestion Pipeline [{mode}] ===\n")

    docs_path = os.path.join(BASE_DIR, "docs")
    documents = load_documents(docs_path)
    chunks = split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    if args.limit:
        chunks = chunks[:args.limit]
        print(f"\n[TEST MODE] Truncated to {len(chunks)} chunks.")

    print(f"\nProcessing {len(chunks)} chunks...")
    create_tigergraph_vector_store(chunks, batch_size=50)
    print("\n✅ Ingestion complete. Chunks and embeddings are stored directly in TigerGraph.")


if __name__ == "__main__":
    main()