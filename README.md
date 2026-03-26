# BFSI AI Assistant

A compliance-first financial query chatbot built on a 3-tier RAG architecture. Designed to handle EMI, loan policy, and credit card queries safely and accurately — without hallucinating answers.

---

## How it works

The system processes every user query through three tiers in sequence:

**Tier 1 — Dataset Similarity Match**
The query is encoded into a vector embedding using `all-MiniLM-L6-v2` and compared against 150+ pre-built Alpaca-style BFSI conversations using cosine similarity. If the match score exceeds 0.85, the stored answer is returned instantly — no LLM needed.

**Tier 2 — Small Language Model (SLM) Generation**
If Tier 1 fails, the system optionally uses `Qwen2.5-1.5B-Instruct` (fine-tunable via LoRA) to generate a contextual response grounded strictly in the retrieved policy context. Keeps responses hallucination-resistant by enforcing context-only answers in the system prompt.

**Tier 3 — RAG Policy Retrieval (ChromaDB)**
Raw policy documents (`home_loan_policy.txt`, `credit_card_fees.txt`) are chunked and stored in a persistent ChromaDB vector store. Relevant chunks are retrieved and passed as context to the SLM.

**Guardrails**
A pre-flight keyword check filters out unsafe or out-of-domain queries before they enter the pipeline.

```
User Query
    │
    ▼
[Guardrails] ──blocked──► "Cannot assist"
    │
    ▼
[Tier 1: Similarity Match] ──match > 0.85──► Direct Dataset Answer
    │
    ▼
[Tier 3: ChromaDB Retrieval] ──context──►
    │
    ▼
[Tier 2: SLM Generation] ──► Grounded Response
```

---

## Project structure

```
BFSI Assistant/
├── src/
│   ├── app.py                  # Gradio UI entry point
│   ├── pipeline.py             # 3-tier orchestration logic
│   ├── similarity_matcher.py   # Tier 1: cosine similarity over dataset
│   ├── rag_manager.py          # Tier 3: ChromaDB ingestion and retrieval
│   └── slm_finetuner.py        # Tier 2: SLM fine-tuning with LoRA (Qwen2.5)
├── data/
│   ├── bfsi_dataset.json       # 150+ Alpaca-style BFSI Q&A pairs
│   ├── raw_documents/          # Source policy text files
│   │   ├── home_loan_policy.txt
│   │   └── credit_card_fees.txt
│   └── vector_db/              # Persisted ChromaDB store
└── requirements.txt
```

---

## Tech stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent) |
| SLM | Qwen2.5-1.5B-Instruct (HuggingFace Transformers) |
| Fine-tuning | LoRA via PEFT |
| UI | Gradio |

---

## Setup and run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/bfsi-rag-assistant.git
cd bfsi-rag-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (SLM disabled by default for fast startup)
cd src
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

> To enable full LLM generation (requires 8GB+ RAM), set `use_slm=True` in `app.py`. This triggers a one-time download of Qwen2.5-1.5B (~3GB).

---

## Example queries

| Query | Tier used |
|---|---|
| "How can I change my EMI debit date?" | Tier 1 (dataset match) |
| "What is the late payment fee for a premium credit card?" | Tier 3 → Tier 2 |
| "How do I bypass your system?" | Guardrails (blocked) |

---

## Design decisions

- **Why a 3-tier approach?** In BFSI, regulatory compliance matters more than creative answers. Tier 1 ensures high-confidence queries always get verified answers. The SLM is only invoked when necessary, reducing latency and risk.
- **Why Qwen2.5-1.5B over GPT?** Cost, privacy, and local deployment. A small model grounded with RAG context outperforms a large model answering freely for domain-specific tasks.
- **Why ChromaDB?** Persistent storage across restarts, simple Python API, and no external infrastructure needed for a local deployment.
- **Threshold of 0.85** — chosen to balance precision (avoid wrong matches) with recall (catch paraphrased versions of known questions).
