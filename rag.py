
import hashlib
import re
from io import BytesIO

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Small, multilingual-compatible English-focused embedding model.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def load_embedding_model():
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)



def extract_text(uploaded_file):
    from io import BytesIO
    from pypdf import PdfReader

    filename = uploaded_file.name
    extension = filename.lower().rsplit(".", 1)[-1]

    try:
        if extension == "pdf":
            reader = PdfReader(BytesIO(uploaded_file.getvalue()))
            pages = []

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({
                        "page": page_number,
                        "text": text.strip(),
                    })

            return pages

        if extension == "txt":
            text = uploaded_file.getvalue().decode(
                "utf-8-sig",
                errors="replace",
            ).strip()

            if not text:
                return []

            return [{"page": 1, "text": text}]

        raise ValueError(
            f"Unsupported file type: {filename}. "
            "Upload a PDF or TXT file."
        )

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not read {filename}: {exc}"
        ) from exc


def split_into_chunks(text, chunk_size=CHUNK_SIZE,
                      overlap=CHUNK_OVERLAP):
    """Split text into overlapping character-based chunks."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def create_document_records(uploaded_files):
    """Extract and chunk documents with source metadata."""
    records = []

    for uploaded_file in uploaded_files:
        pages = extract_text(uploaded_file)

        for page_data in pages:
            chunks = split_into_chunks(page_data["text"])

            for chunk_index, chunk in enumerate(chunks):
                records.append(
                    {
                        "text": chunk,
                        "source": uploaded_file.name,
                        "page": page_data["page"],
                        "chunk_index": chunk_index,
                    }
                )

    return records



def index_documents(records, embedding_model):
    """Replace the active document index with the supplied records."""
    import chromadb

    if not records:
        raise ValueError(
            "No readable document text was found. "
            "Upload a PDF or TXT file containing text."
        )

    # Use one active collection rather than creating a new
    # collection for every upload batch.
    client = chromadb.PersistentClient(path="./chroma_db")

    collection_name = "agentassist_documents"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare all embeddings before removing the previous index.
    prepared = []

    for start in range(0, len(records), 64):
        batch = records[start:start + 64]
        texts = [record["text"] for record in batch]

        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()

        prepared.append((batch, embeddings))

    # Clear the old active index so removed documents cannot
    # remain searchable after a replacement upload.
    existing = collection.get()
    existing_ids = existing.get("ids", [])

    for start in range(0, len(existing_ids), 500):
        collection.delete(ids=existing_ids[start:start + 500])

    # Insert the new document set.
    for batch, embeddings in prepared:
        ids = []
        documents = []
        metadatas = []

        for record in batch:
            source = record["source"]
            page = record["page"]
            chunk_index = record["chunk_index"]

            chunk_key = (
                f"{source}|{page}|{chunk_index}|{record['text']}"
            )
            chunk_id = hashlib.sha256(
                chunk_key.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)
            documents.append(record["text"])
            metadatas.append({
                "source": source,
                "page": int(page),
                "chunk_index": int(chunk_index),
            })

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return client, collection


def retrieve_relevant_chunks(query, collection, embedding_model,
                             top_k=4):
    """Retrieve semantically similar document chunks."""
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append(
            {
                "text": document,
                "source": metadata["source"],
                "page": metadata["page"] or None,
                "distance": distance,
            }
        )

    return matches
