
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from rag import retrieve_relevant_chunks
from sentence_transformers import SentenceTransformer


def test_retrieval_returns_relevant_document():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    client = chromadb.Client()
    collection = client.create_collection(
        name="retrieval_test",
        metadata={"hnsw:space": "cosine"},
    )

    documents = [
        "AgentAssist uses ChromaDB for vector storage.",
        "The prototype supports document upload and semantic search.",
        "The weather in Paris is often mild in spring.",
    ]

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=["d1", "d2", "d3"],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {"source": "test.txt", "page": 1},
            {"source": "test.txt", "page": 1},
            {"source": "other.txt", "page": 1},
        ],
    )

    results = retrieve_relevant_chunks(
        "Which vector database does AgentAssist use?",
        collection,
        model,
        top_k=2,
    )

    assert len(results) == 2
    assert "ChromaDB" in results[0]["text"]
