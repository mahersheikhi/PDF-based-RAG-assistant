"""Interactive PDF-based RAG assistant."""
from pathlib import Path
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_PATH = Path("company_policies_demo.pdf")
USE_LLM = False
OLLAMA_MODEL = "qwen2.5:3b"

def read_pdf(path):
    chunks, metadata = [], []
    for page, item in enumerate(PdfReader(str(path)).pages, 1):
        text = (item.extract_text() or "").strip()
        if text:
            chunks.append(text)
            metadata.append({"source": path.name, "page": page})
    return chunks, metadata

def ask_ollama(question, documents, sources):
    import requests
    context = "\n\n".join(f"[Source: {m['source']}, page {m['page']}]\n{d}" for d, m in zip(documents, sources))
    prompt = f"""Answer the employee question using only the source text. Cite the page. Say 'I do not have sufficient information about this.' only when no source answers it.

Source text:
{context}

Employee question: {question}
Answer:"""
    response = requests.post("http://host.docker.internal:11434/api/generate", json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=120)
    response.raise_for_status()
    return response.json()["response"].strip()

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError("Run python create_demo_pdf.py first.")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = chromadb.PersistentClient(path="./chroma_data").get_or_create_collection("company_policy_pdf")
    chunks, metadata = read_pdf(PDF_PATH)
    collection.upsert(ids=[f"page_{i}" for i in range(1, len(chunks)+1)], documents=chunks, metadatas=metadata, embeddings=model.encode(chunks, normalize_embeddings=True).tolist())
    print("Employee policy assistant ready. Type exit to quit.")
    while True:
        question = input("Employee question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = collection.query(query_embeddings=model.encode([question], normalize_embeddings=True).tolist(), n_results=min(2, len(chunks)), include=["documents", "metadatas"])
        docs, sources = result["documents"][0], result["metadatas"][0]
        if USE_LLM:
            print("\nAnswer:\n" + ask_ollama(question, docs, sources))
        else:
            print("\nRetrieved policy text:")
            for doc, source in zip(docs, sources):
                print(f"- {source['source']}, page {source['page']}: {doc}")

if __name__ == "__main__":
    main()
