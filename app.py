import os
import gradio as gr

from src.utils import (
    load_data,
    get_text_chunks,
    get_vector_store,
    build_qa_chain,
    ask,
)

# ---- Config ----
DATA_FILE = os.getenv("RAG_DATA_FILE", "/home/ims24/RAG/eleven_madison_park_data.txt")
PERSIST_DIR = os.getenv("CHROMA_DIR", "/home/ims24/RAG/research/chroma_db")

# ---- Build / load everything once at startup ----
docs = load_data(DATA_FILE)
chunks = get_text_chunks(docs)

# NOTE: make sure your get_vector_store supports persist_directory
# (recommended signature: get_vector_store(text_chunks, persist_directory="chroma_db"))
vector_store = get_vector_store(chunks)

qa_chain = build_qa_chain(
    vector_store,
    k=3,
    temperature=0.2,
    verbose=False,
)

def rag_answer(question: str, k: int = 3, temperature: float = 0.2, show_snippets: bool = False):
    """Gradio callback"""
    if not question or not question.strip():
        return "Ask a question above.", ""

    # rebuild retriever/chain only if user changes k/temp (simple approach)
    local_chain = build_qa_chain(vector_store, k=int(k), temperature=float(temperature), verbose=False)

    answer, sources, source_docs = ask(local_chain, question)

    extra = []
    extra.append(f"**Sources:** {sources or 'None'}")

    if show_snippets and source_docs:
        extra.append("\n**Snippets:**")
        for i, doc in enumerate(source_docs, 1):
            snippet = doc.page_content[:250].strip().replace("\n", " ")
            extra.append(f"{i}. {snippet}")

    return answer, "\n".join(extra)

with gr.Blocks(title="RAG Q&A") as demo:
    gr.Markdown("# RAG Q&A (Chroma + LangChain)")
    gr.Markdown(f"Using data file: `{DATA_FILE}` | Persist dir: `{PERSIST_DIR}`")

    question = gr.Textbox(label="Your question", placeholder="Ask something...", lines=2)

    with gr.Row():
        k = gr.Slider(1, 10, value=3, step=1, label="Top-k documents")
        temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.1, label="Temperature")

    show_snippets = gr.Checkbox(value=False, label="Show source snippets")

    btn = gr.Button("Ask")
    answer_out = gr.Textbox(label="Answer", lines=6)
    meta_out = gr.Markdown()

    btn.click(
        fn=rag_answer,
        inputs=[question, k, temperature, show_snippets],
        outputs=[answer_out, meta_out],
    )

if __name__ == "__main__":
    demo.launch()