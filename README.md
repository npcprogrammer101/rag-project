# PDF RAG Agent

A fully local, **100% free** Retrieval-Augmented Generation (RAG) agent that answers questions about any PDF you give it — running entirely on your own machine with no API keys, no cloud, and no usage limits.

Ask questions in plain language and get answers grounded *only* in the contents of your document, delivered through a polished terminal interface with streaming responses, markdown rendering, and source inspection.

---

## What it does

Point it at a PDF, and it builds a searchable knowledge base from the document. When you ask a question, it retrieves the most relevant passages and feeds them to a local language model, which answers using only what's actually in the document — and tells you honestly when something isn't there, rather than making things up.

Everything runs locally via [Ollama](https://ollama.com), so your documents never leave your machine and there's nothing to pay for.

---

## Features

- **Fully local & free** — no API keys, no cloud services, no usage limits. Runs on your own hardware.
- **Private** — your documents and questions never leave your machine.
- **Grounded answers** — responses are based only on the PDF's content, with an explicit "I couldn't find that" fallback to discourage hallucination.
- **Streaming responses** — answers appear token-by-token as they're generated, then render as clean formatted markdown.
- **Polished CLI** — a `rich` + `prompt_toolkit` interface with a thinking spinner, command history, and autocomplete.
- **Source inspection** — the `/sources` command shows exactly which chunks were retrieved for your last question, making it easy to debug retrieval quality.
- **Persistent vector store** — the document is embedded once and saved to disk, so subsequent runs start instantly.

---

## How it works

The agent follows the standard RAG pipeline:

```
PDF → Load → Chunk → Embed → Store (Chroma) → Retrieve → Generate (LLM) → Answer
```

1. **Load** — the PDF is read page by page with `PyPDFLoader`.
2. **Chunk** — text is split into overlapping segments with `RecursiveCharacterTextSplitter` (1500 characters, 300-character overlap) so related content stays together.
3. **Embed** — each chunk is converted into a vector using a local embedding model (`nomic-embed-text`).
4. **Store** — vectors are persisted to a local [Chroma](https://www.trychroma.com/) database (`./chroma_db`).
5. **Retrieve** — at query time, the most relevant chunks are pulled from the store.
6. **Generate** — the retrieved context plus your question are passed to a local LLM (`llama3.1:8b`) via a grounding prompt that restricts answers to the supplied context.

---

## Tech stack

| Component      | Tool                                              |
| -------------- | ------------------------------------------------- |
| Orchestration  | [LangChain](https://www.langchain.com/) (LCEL)    |
| LLM            | `llama3.1:8b` via [Ollama](https://ollama.com)    |
| Embeddings     | `nomic-embed-text` via Ollama                     |
| Vector store   | [Chroma](https://www.trychroma.com/) (local, persistent) |
| Interface      | [rich](https://github.com/Textualize/rich) + [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) |

---

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com)** installed and running

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/npcprogrammer101/rag-project.git
cd rag-project
```

**2. Set up a virtual environment** (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
```

**3. Install Python dependencies**

```bash
pip install langchain langchain-community langchain-chroma \
    langchain-huggingface langchain-ollama pypdf sentence-transformers \
    rich prompt_toolkit
```

**4. Pull the Ollama models**

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

---

## Usage

**1. Add your PDF**

Place your PDF in the project (e.g. in a `sample_files/` folder) and update the path near the top of the script:

```python
PDF_PATH = "./sample_files/your_document.pdf"
```

**2. Run the agent**

```bash
python agent.py
```

On the **first run**, the agent embeds the PDF and builds the vector store (this takes a moment). On **subsequent runs**, it loads the saved store and starts instantly.

**3. Ask questions**

```
You › What are the main topics covered in this document?
```

---

## Commands

While the agent is running, these slash-commands are available:

| Command     | Description                                          |
| ----------- | ---------------------------------------------------- |
| `/help`     | Show the list of commands                            |
| `/sources`  | Show the chunks retrieved for your last question     |
| `/model`    | Show which models and settings are in use            |
| `/clear`    | Clear the screen                                     |
| `/quit`     | Exit the agent                                       |

---

## Configuration

Key settings live at the top of the script and can be tuned to your needs:

```python
PDF_PATH    = "./sample_files/mongodb.pdf"   # the document to query
DB_DIR      = "./chroma_db"                   # where the vector store is saved
EMBED_MODEL = "nomic-embed-text"              # local embedding model
LLM_MODEL   = "llama3.1:8b"                   # local language model
K           = 10                              # number of chunks to retrieve
```

**A few tuning notes:**

- **`K`** controls how many chunks are retrieved per question. Higher values give broader coverage but a larger prompt (and slower answers). For a single document, somewhere between 6 and 10 is a good range.
- **Chunk size / overlap** are set in `RecursiveCharacterTextSplitter`. Larger chunks keep long passages and lists intact; smaller chunks give more precise retrieval of scattered facts.
- **Changing the embedding model or chunk settings requires rebuilding the store.** Delete the `./chroma_db` folder and run again, since the embedding model must match between building and querying.

```bash
rm -rf ./chroma_db
```

---

## Troubleshooting

**The agent is slow to answer.** Generation speed depends on your hardware and the LLM size. On machines with limited RAM, try a smaller model such as `qwen2.5:3b`, and confirm Ollama is using your GPU with `ollama ps`.

**An answer only contains part of what I expected.** This is usually a retrieval issue — the relevant content wasn't pulled. Use `/sources` to see what was retrieved, then try raising `K` or increasing the chunk size and overlap (and rebuild the store).

**`Collection expecting embedding with dimension of X, got Y`.** You changed the embedding model but kept the old vector store. Delete `./chroma_db` and run again to rebuild it with the new model.

**`Error: ... Is Ollama running?`** Make sure the Ollama app or service is running. You can start it manually with `ollama serve`.

---

## How "free" works here

Unlike RAG setups that call a hosted API for embeddings or generation, every model in this project runs locally through Ollama:

- **No API keys** to obtain or manage.
- **No usage limits** — ask as many questions as you like.
- **No data leaves your machine** — fully private.

The only costs are your own hardware resources (RAM, disk, and compute time). The trade-off is that a local model is less powerful than a frontier hosted model, but for grounded question-answering over a single document, it performs very well.

---

## Possible extensions

This project is a solid foundation that can grow in several directions:

- **Multi-format support** — add loaders for Word, Markdown, CSV, and PowerPoint files.
- **Multilingual querying** — swap in a multilingual embedding model (e.g. a `qwen3` embedding) to ask questions in one language about documents in another.
- **MMR retrieval** — use Maximal Marginal Relevance for more diverse chunk coverage on "list everything about X" questions.
- **Reranking** — add a cross-encoder reranker to improve retrieval precision.
- **Hybrid search** — combine vector search with keyword (BM25) search for exact-match queries.

---

## License

This project is provided as-is for learning and personal use.