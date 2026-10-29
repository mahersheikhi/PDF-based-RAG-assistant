# PDF Policy RAG Assistant

A small, local RAG demo that answers employee questions from a company-policy PDF. It uses pypdf to extract text, SentenceTransformers to create embeddings, and ChromaDB for semantic retrieval.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python create_demo_pdf.py
python rag_pdf_assistant.py
```

Example questions: `How do I reset my password?` and `How many vacation days do I receive?`

## Optional local LLM

Run `ollama pull qwen2.5:3b`, then set `USE_LLM = True`. Answers are constrained to retrieved policy text and cite the source page.

## Docker

```powershell
docker build -t pdf-policy-rag-assistant .
docker run -it --rm -v "${PWD}/chroma_data:/app/chroma_data" pdf-policy-rag-assistant
```
