# Architecture

## Components

| Component | Responsibility |
|---|---|
| `ragqa/ingestion/loaders.py` | Load txt/md files, extract PDF text per page (PyMuPDF/pypdf), OCR hook |
| `ragqa/ingestion/cleaning.py` | Whitespace normalization, de-hyphenation, paragraph splitting |
| `ragqa/ingestion/chunking.py` | Word-boundary windows with overlap, source/page/offset metadata |
| `ragqa/ingestion/pipeline.py` | Ingestion pipeline + document metadata store (status, versions, roles) |
| `ragqa/embeddings.py` | Provider interface: hashing (default), sentence-transformers, OpenAI |
| `ragqa/store.py` | Vector store interface: exact in-memory cosine, FAISS adapter |
| `ragqa/retrieval.py` | Query embedding, hybrid vector+keyword scoring, threshold, reranking |
| `ragqa/generation.py` | Context builder (chunks labeled with sources), extractive + OpenAI answerers |
| `ragqa/service.py` | `QAService`: the query pipeline end to end, abstention rule |
| `ragqa/api.py` | FastAPI service: health, ingest, list, ask |
| `ragqa/cli.py` | CLI: ingest, ask, serve, demo |

## Ingestion pipeline (runs when documents are added or updated)

```
Documents ──► Text Extraction ──► Cleaning & Chunking ──► Embeddings ──► Vector DB
                │                    │                      │            │
                │                    └── metadata: source,  │            └── chunk
                │                        page, offsets      │                embeddings
                └── PDFs via PyMuPDF/pypdf;                 └── same provider
                    scanned pages -> OCR hook                    as the query side
                    (fail-closed, never silent)

Document Metadata Store (in the pipeline):
  doc_id, title, source, status (pending/indexed/failed),
  version (bumped on re-ingest), allowed_roles (empty = public)
```

**Chunking.** Character windows (default 800, overlap 150) cut on word
boundaries. Overlap exists so an answer span straddling a boundary is
still retrievable whole from at least one chunk. For PDFs, each page is
chunked separately so the page number on every chunk is exact — that is
what makes citations checkable.

**Re-ingest replaces, never duplicates.** Re-ingesting a doc_id deletes
its old chunks, bumps its version, and re-indexes. Stale chunks are the
quietest source of wrong answers in a long-lived RAG system.

## Query pipeline (runs when a user asks a question)

```
User Question ──► Query Embedding ──► Retrieve Relevant Chunks ──► Build Context
                    (same model         (vector similarity +     (top-k chunks,
                     as ingestion)       optional keyword blend,   each labeled
                                         threshold, rerank,       with its source)
                                         access filter)
                                                              │
                                              Generate Answer ──► Response
                                              (extractive quotes  (answer +
                                               or LLM instructed   citations +
                                               to cite sources)    document links)
```

**Retrieval order of operations.** (1) Embed with the same provider as
ingestion. (2) Over-fetch (3× top-k) so later filtering still leaves
top-k. (3) Drop chunks the caller's roles may not see. (4) Blend
vector similarity with keyword term coverage (default weight 0.3).
(5) Drop everything below the similarity threshold. (6) Rerank by
blended score, then term coverage, then vector score.

**Abstention rule.** An empty retrieval result produces an explicit
"I couldn't find a relevant answer in the indexed documents" response,
with no citations and `abstained: true`. The answerer never sees
irrelevant chunks, so it cannot dress up a guess as an answer.

**Citations.** Every retrieved chunk becomes a citation (source, page,
excerpt) when the context is built. The extractive answerer quotes
sentences from those chunks; the OpenAI adapter receives the same
labeled context with instructions to cite every claim and to say when
the context does not contain the answer.

## Failure modes this design addresses

| Failure | Mitigation |
|---|---|
| Model upgrade invalidates stored vectors | One provider resolved from settings for both sides; re-index on provider change |
| Answer stitched from the wrong chunk | Hybrid scoring + reranking on term coverage; citations expose the source |
| Hallucination on out-of-corpus questions | Threshold + explicit abstention |
| Restricted documents leaking into answers | Role filter at retrieval, before the answerer sees anything |
| Scanned pages silently missing | Fail-closed OCR hook; document marked failed, not half-indexed |
| Stale answers after a document update | Re-ingest replaces chunks and bumps the version |

## Deployment

Stateless core: the in-memory store suits demos and small corpora. For
production, persist the vector store behind FAISS (adapter included) or
a managed vector database, and the metadata store behind a real
database; the `VectorStore` interface and the pipeline's document dict
are the seams. The Dockerfile runs the API; wire `uvicorn` with
`create_app()` for the HTTP service. CI is defined in
`docs/github-actions-ci.yml` — activate it by moving that file to
`.github/workflows/ci.yml` (the push integration cannot create
`.github/workflows/` directly).
