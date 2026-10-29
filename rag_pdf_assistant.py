"""Interactive PDF-based RAG assistant."""
from pathlib import Path
import os
import re

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_PATH = Path("company_policies_demo.pdf")
USE_LLM = True
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
POLICY_HEADINGS = ("IT and password policy", "Vacation policy", "Remote-work policy", "Expense policy")

def split_policy_text(text):
    heading_pattern = "|".join(re.escape(heading) for heading in POLICY_HEADINGS)
    sections = re.split(f"(?=^{heading_pattern}$)", text, flags=re.MULTILINE)
    return [section.strip() for section in sections if any(section.strip().startswith(heading) for heading in POLICY_HEADINGS)]

def read_pdf(path):
    chunks, metadata = [], []
    for page, item in enumerate(PdfReader(str(path)).pages, 1):
        for section in split_policy_text((item.extract_text() or "").strip()):
            chunks.append(section)
            metadata.append({"source": path.name, "page": page, "topic": section.splitlines()[0]})
    if not chunks:
        raise ValueError("No readable policy text was found in the PDF.")
    return chunks, metadata

def show_sources(documents, sources):
    print("\nRetrieved policy text:")
    for document, source in zip(documents, sources):
        print(f"- {source['topic']} ({source['source']}, page {source['page']}):\n  {document}")

def retrieved_answer(documents, sources):
    source = sources[0]
    return f"Based on the {source['topic']} ({source['source']}, page {source['page']}):\n{documents[0]}"

def ask_ollama(question, documents, sources):
    import requests
    context = "\n\n".join(f"<source topic=\"{s['topic']}\" page=\"{s['page']}\">\n{d}\n</source>" for d, s in zip(documents, sources))
    prompt = f"""You answer employee policy questions. Use ONLY the source text in the <source> tags. A source is authoritative. Give one short, direct answer and end with "(Source: page N)". Never say information is insufficient when a source answers the question. Only say "I do not have sufficient information about this." when no source is relevant.

Employee question: {question}

Retrieved source text:
{context}

Answer:"""
    response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}, timeout=120)
    response.raise_for_status()
    answer = response.json()["response"].strip()
    return retrieved_answer(documents, sources) if "do not have sufficient information" in answer.lower() else answer

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError("Run python create_demo_pdf.py first.")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_data")
    client.delete_collection("company_policy_pdf")
    collection = client.get_or_create_collection("company_policy_pdf")
    chunks, metadata = read_pdf(PDF_PATH)
    collection.upsert(ids=[f"policy_{index}" for index in range(len(chunks))], documents=chunks, metadatas=metadata, embeddings=model.encode(chunks, normalize_embeddings=True).tolist())
    print("Employee policy assistant ready. Type exit to quit.")
    while True:
        question = input("Employee question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = collection.query(query_embeddings=model.encode([question], normalize_embeddings=True).tolist(), n_results=min(2, len(chunks)), include=["documents", "metadatas"])
        documents, sources = result["documents"][0], result["metadatas"][0]
        if USE_LLM:
            try:
                print("\nAnswer:\n" + ask_ollama(question, documents, sources))
            except Exception as error:
                print(f"\nCould not reach Ollama: {error}")
                show_sources(documents, sources)
        else:
            show_sources(documents, sources)

if __name__ == "__main__":
    main()
