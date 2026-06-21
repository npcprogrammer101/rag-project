"""
PDF RAG Agent — fully free, runs locally, with a polished CLI.

Loads a PDF, embeds it into a local Chroma vector store, and answers
questions using only the content of that PDF via a local Ollama LLM.
Wrapped in a rich + prompt_toolkit interface: streaming answers, a thinking
spinner, markdown rendering, slash-commands, and source inspection.

Requirements:
    pip install langchain langchain-community langchain-chroma \
        langchain-huggingface langchain-ollama pypdf sentence-transformers \
        rich prompt_toolkit

    Install Ollama from https://ollama.com, then:
        ollama pull llama3.1:8b
        ollama pull nomic-embed-text
"""

import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.rule import Rule

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style as PTStyle

# ---- Config ----
PDF_PATH = "./sample_files/mongodb.pdf"   # <-- put your PDF path here
DB_DIR = "./chroma_db"                     # where the vector store persists
EMBED_MODEL = "nomic-embed-text"           # local, free embeddings
LLM_MODEL = "llama3.1:8b"                  # local, free LLM via Ollama
K = 10                                     # chunks to retrieve

console = Console()

# Slash-commands available in the session
COMMANDS = {
    "/help": "Show this help message",
    "/sources": "Show the chunks retrieved for your last question",
    "/clear": "Clear the screen",
    "/model": "Show which models are in use",
    "/quit": "Exit the agent",
}


# ---------- RAG setup (unchanged logic) ----------

def build_or_load_vectorstore(embeddings):
    """Build the vector store from the PDF on first run; load it after."""
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        console.print("[dim]Loading existing vector store...[/dim]")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    console.print("[yellow]Building vector store from PDF (first run)...[/yellow]")
    if not os.path.exists(PDF_PATH):
        console.print(f"[red]PDF not found at '{PDF_PATH}'. Edit PDF_PATH at the top of the file.[/red]")
        sys.exit(1)

    docs = PyPDFLoader(PDF_PATH).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )
    chunks = splitter.split_documents(docs)
    console.print(f"[dim]Split into {len(chunks)} chunks.[/dim]")

    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=DB_DIR
    )
    return vectorstore


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant answering questions about a document.
Use ONLY the context below to answer. If the answer is not in the context,
say "I couldn't find that in the document." When the source presents a list
or numbered sequence, preserve its original order and numbering. Do not
make things up.

Context:
{context}

Question: {question}

Answer:"""
    )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ---------- CLI helpers ----------

def print_banner():
    title = Text("PDF RAG Agent", style="bold cyan")
    subtitle = Text(f"\nAsk questions about: {os.path.basename(PDF_PATH)}", style="dim")
    hint = Text("\nType /help for commands, /quit to exit.", style="dim italic")
    console.print(Panel(Text.assemble(title, subtitle, hint), border_style="cyan", padding=(1, 2)))


def print_help():
    console.print(Rule("[bold]Commands[/bold]", style="cyan"))
    for cmd, desc in COMMANDS.items():
        console.print(f"  [bold cyan]{cmd:<10}[/bold cyan] [dim]{desc}[/dim]")
    console.print(Rule(style="cyan"))


def show_sources(docs):
    """Display the retrieved chunks so you can see what the model was given."""
    if not docs:
        console.print("[dim]No sources yet — ask a question first.[/dim]")
        return
    console.print(Rule("[bold]Retrieved chunks[/bold]", style="magenta"))
    for i, d in enumerate(docs):
        page = d.metadata.get("page", "?")
        preview = d.page_content.strip().replace("\n", " ")[:240]
        console.print(
            Panel(
                f"[dim]{preview}...[/dim]",
                title=f"[magenta]chunk {i + 1} · page {page}[/magenta]",
                border_style="magenta",
                padding=(0, 1),
            )
        )


def stream_answer(chain, question):
    """Stream the answer token-by-token under a thinking spinner, then
    render the finished answer as markdown."""
    collected = ""
    spinner = Spinner("dots", text=" Thinking...", style="cyan")

    # Phase 1: spinner until the first token arrives, then live raw text.
    with Live(spinner, console=console, refresh_per_second=12, transient=True) as live:
        first_token_seen = False
        for chunk in chain.stream(question):
            collected += chunk
            if not first_token_seen and chunk.strip():
                first_token_seen = True
            # Show the text streaming in (plain) while generating.
            live.update(Text(collected, style="white"))

    # Phase 2: re-render the completed answer as nicely formatted markdown.
    console.print(Panel(Markdown(collected), title="[green]Agent[/green]",
                        border_style="green", padding=(1, 2)))
    return collected


# ---------- main ----------

def main():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = build_or_load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": K})
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    chain = build_chain(retriever, llm)

    console.clear()
    print_banner()

    # prompt_toolkit session: history (up-arrow) + slash-command autocomplete
    completer = WordCompleter(list(COMMANDS.keys()), sentence=True)
    session = PromptSession(
        history=InMemoryHistory(),
        completer=completer,
        style=PTStyle.from_dict({"prompt": "bold ansicyan"}),
    )

    last_sources = []

    while True:
        try:
            question = session.prompt("\nYou › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue

        # Handle slash-commands
        if question.startswith("/"):
            cmd = question.lower().split()[0]
            if cmd in {"/quit", "/exit", "/q"}:
                console.print("[dim]Goodbye.[/dim]")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/clear":
                console.clear()
                print_banner()
            elif cmd == "/model":
                console.print(f"[cyan]LLM:[/cyan] {LLM_MODEL}   "
                              f"[cyan]Embeddings:[/cyan] {EMBED_MODEL}   "
                              f"[cyan]k:[/cyan] {K}")
            elif cmd == "/sources":
                show_sources(last_sources)
            else:
                console.print(f"[red]Unknown command:[/red] {cmd}  "
                              f"[dim](try /help)[/dim]")
            continue

        # Capture sources for /sources, then stream the answer
        last_sources = retriever.invoke(question)
        try:
            stream_answer(chain, question)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            console.print("[dim]Is Ollama running? Try `ollama serve` in another terminal.[/dim]")


if __name__ == "__main__":
    main()