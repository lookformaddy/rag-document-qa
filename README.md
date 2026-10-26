# rag-document-qa — Document Question Answering with RAG

Ask questions in plain English over a collection of documents — PDFs,
docs, internal knowledge-base articles, web pages — and get answers
grounded in those documents, with references to the relevant sources.

This is not a chatbot that sends a question to an LLM and displays
whatever comes back. It is a **data pipeline combined with a retrieval
system and a language model**, built around the classic two-pipeline
RAG architecture.

## The enterprise problem

Enterprise knowledge lives in documents, but keyword search returns
documents, not answers — and raw LLM answers hallucinate. A RAG system
closes both gaps: retrieval narrows the LLM to the passages that
actually contain the answer, and citations let a reader verify every
claim against its source. Done carelessly, though, RAG fails in
predictable ways: a model upgrade silently invalidates stored vectors,
an answer is stitched from the wrong chunk, restricted documents leak
into answers, and the system invents an answer when the documents say
nothing. This project addresses each of those failure modes explicitly.

## Architecture overview

```
INGESTION PIPELINE  (runs when documents are added or updated)
Documents ──► Text Extraction ──► Cleaning & Chunking ──► Embeddings ──► Vector DB
(PDFs, docs,        (PyMuPDF/pypdf,    (overlap, metadata:        (same model
 KB, web pages)      OCR hook)          source, page)              as query)
                                                              │
                                              Document Metadata Store
                                              (info, status, versions,
                                               access control)

QUERY PIPELINE  (runs when a user asks a question)
User Question ──► Query Embedding ──► Retrieve Relevant Chunks ──► Build Context
                    (same model         (vector similarity + optional
                     as ingestion)       keyword search + reranking)
                                                              │
                                              Generate Answer ──► Response
                                              (LLM instructed   (answer +
                                               to cite sources)  citations +
                                                                 document links)
```

See [`docs/architecture.md`](docs/architecture.md) for the full design,
including the failure modes, the abstention rule, and the upgrade
playbook.

## How to run

Core runs on the standard library only — no API keys, no downloads:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # API + test deps; core itself is stdlib-only
python examples/run_demo.py              # ingest sample docs, ask 3 questions
```

The demo answers two questions from the synthetic sample documents and
abstains on a third ("What is the capital of France?") because the
documents contain no relevant passage — abstention, not hallucination.

CLI:

```bash
python -m ragqa.cli ingest examples/sample_docs/*.txt
python -m ragqa.cli ask "What is the refund window for standard orders?"
```

API (needs `pip install -r requirements.txt` plus the api extra):

```bash
pip install ".[api]"
python -m ragqa.cli serve                 # FastAPI on 127.0.0.1:8000
curl localhost:8000/health
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "What is the refund window for standard orders?"}'
```

Endpoints: `GET /health`, `POST /documents`, `GET /documents`,
`POST /ask` (accepts `roles` for access-filtered retrieval).

Optional providers, all behind env flags (`RAGQA_EMBEDDING_PROVIDER`,
`RAGQA_LLM_PROVIDER`):

| Provider | Install | Notes |
|---|---|---|
| Hashing embedder (default) | — | Deterministic, offline, zero keys |
| sentence-transformers | `pip install ".[st]"` | Real semantic vectors |
| OpenAI embeddings / chat | `OPENAI_API_KEY=...` | Via `urllib`, no SDK needed |
| FAISS vector store | `pip install ".[faiss]"` | For larger corpora |

## Key design decisions and trade-offs

1. **Same model on both sides, enforced by construction.** Ingestion
   and query resolve their embedder from one settings object. Vectors
   from different models are not comparable, and mixing them is the
   quietest way to break a RAG system.
2. **Hybrid retrieval by default.** Pure vector search misses exact
   terms (policy numbers, thresholds); pure keyword search misses
   paraphrase. Blending vector similarity with term coverage, then
   reranking, beats either alone on document QA.
3. **Abstention is a retrieval decision.** If nothing passes the
   similarity threshold, the answerer says so explicitly. A RAG system
   that never says "I don't know" will eventually invent an answer.
4. **Extractive answering as the offline default.** It can only quote
   retrieved sentences, so it cannot hallucinate new facts. The
   OpenAI adapter swaps in for fluent answers with the same citation
   contract — same context, same sources, same abstention instruction.
5. **Citations carry source + page + excerpt.** A citation that does
   not let the reader find the passage is decoration, not evidence.
6. **Access control at retrieval, not in the store.** Documents carry
   allowed roles; the retriever filters before ranking, so restricted
   content never reaches the answerer — or the prompt.
7. **Fail-closed OCR and extraction.** A scanned page with no OCR
   backend fails the document loudly instead of silently dropping its
   content from the index.

## What it demonstrates

Two-pipeline RAG architecture end to end; chunking with overlap and
page-aware metadata; pluggable embedding providers with an offline
default; exact and FAISS vector stores behind one interface; hybrid
retrieval with reranking; citation-attached generation with explicit
abstention; role-based access filtering; a FastAPI service and CLI over
the same core — all test-covered and runnable with zero API keys.

## Clean-room note

All code is original. All sample documents are synthetic, generic
fictional policies (refund, remote work, deployment runbook). No
employer code, data, internal architecture, project names, or
proprietary designs are used anywhere in this repository.

## CI

The CI definition ships as `docs/github-actions-ci.yml` because the
push integration cannot create `.github/workflows/` directly. Activate
it by moving that file to `.github/workflows/ci.yml`.
