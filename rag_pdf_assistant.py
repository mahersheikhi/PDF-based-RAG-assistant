"""Interactive PDF-based RAG assistant."""
from pathlib import Path
import os

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_PATH = Path("company_policies_demo.pdf")
USE_LLM = True
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")


def read_pdf(path):
    chunks, metadata = [], []
    for page, item in enumerate(PdfReader(str(path)).pages, 1):
        text = (item.extract_text() or "").strip()
        if text:
            chunks.append(text)
            metadata.append({"source": path.name, "page": page})
    return chunks, metadata


def show_sources(documents, sources):
    print("\nRetrieved policy text:")
    for document, source in zip(documents, sources):
        print(f"- {source['source']}, page {source['page']}: {document}")


def ask_ollama(question, documents, sources):
    """Generate a source-grounded answer with a local Ollama model."""
    import requests

    context = "\n\n".join(
        f"[Source: {source['source']}, page {source['page']}]\n{document}"
        for document, source in zip(documents, sources)
    )
    prompt = f"""You are a helpful employee policy assistant.
Answer using only the source text below. Give a concise direct answer and cite
the source page. Only say "I do not have sufficient information about this."
when no source answers the question.

Source text:
{context}

Employee question: {question}

Answer:"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError("Run python create_demo_pdf.py first.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = chromadb.PersistentClient(path="./chroma_data").get_or_create_collection(
        "company_policy_pdf"
    )
    chunks, metadata = read_pdf(PDF_PATH)
    collection.upsert(
        ids=[f"page_{index}" for index in range(1, len(chunks) + 1)],
        documents=chunks,
        metadatas=metadata,
        embeddings=model.encode(chunks, normalize_embeddings=True).tolist(),
    )

    print("Employee policy assistant ready. Type exit to quit.")

    while True:
        question = input("Employee question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = collection.query(
            query_embeddings=model.encode(
                [question], normalize_embeddings=True
            ).tolist(),
            n_results=min(2, len(chunks)),
            include=["documents", "metadatas"],
        )
        documents = result["documents"][0]
        sources = result["metadatas"][0]

        if USE_LLM:
            try:
                print("\nAnswer:\n" + ask_ollama(question, documents, sources))
            except Exception as error:
                print(f"\nCould not reach Ollama: {error}")
                print("Showing the retrieved policy text instead.")
                show_sources(documents, sources)
        else:
            show_sources(documents, sources)


if __name__ == "__main__":
    main()
