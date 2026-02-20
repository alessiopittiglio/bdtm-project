import gradio as gr
import torch

from ragqa import config
from ragqa.embedding_utils import encode_texts, get_embedding_model
from ragqa.llm_interface import load_model, generate_response
from ragqa.rag_core import create_client, get_or_create_collection, query_collection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLLECTION_NAME = "mpnet_256_clean_with_metadata"
EMBEDDING_NAME = "sentence-transformers/all-mpnet-base-v2"

MODEL_PATH = config.MODELS_DIR / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

MODEL_CONFIG = {
    "n_ctx": 8192,
    "n_gpu_layers": 23,
    "flash_attn": True,
    "verbose": False,
}

GENERATION_CONFIG = {
    "temperature": 0.7,
}

embedding_model = get_embedding_model(EMBEDDING_NAME, DEVICE)
llm = load_model(MODEL_PATH, config=MODEL_CONFIG)

client = create_client(config.VECTOR_STORE_DIR)
collection = get_or_create_collection(client, COLLECTION_NAME)


def build_context_block(docs, metas, include_metadata):
    """Create a structured context block for the LLM prompt."""
    sections = []

    for idx, (doc, meta) in enumerate(zip(docs, metas), start=1):
        lines = [f"CONTEXT {idx}"]

        if include_metadata and meta:
            lines.extend(f"{k}: {v}" for k, v in meta.items())

        lines.append(doc)
        sections.append("\n".join(lines))

    return "\n\n".join(sections) + "\n\nEND CONTEXT\n"


def build_filter(course, lecture, professor, date, module=None):
    """Build metadata filter for vector search."""
    filters = {
        "course": course,
        "lecture_num": lecture,
        "instructor": professor,
        "lecture_date": date,
        "module": module,
    }

    where = {k: v for k, v in filters.items() if v}
    return where or None


def embed_question(question):
    """Encode question into embedding."""
    embedding = encode_texts([question], embedding_model, normalize=True)[0]
    return embedding.tolist()


def retrieve_documents(question_embedding, top_k, metadata_filter):
    """Retrieve documents from the vector store."""
    results = query_collection(
        collection,
        query_embeddings=[question_embedding],
        n_results=top_k,
        metadata_filter=metadata_filter,
    )

    if results is None:
        raise RuntimeError("Retrieval failed.")

    return results["documents"][0], results["metadatas"][0]


def build_prompt(question, context_block):
    """Create final prompt for the LLM."""
    return f"""
{context_block}

QUESTION: {question}
ANSWER:
"""


def format_retrieval_output(docs, metas):
    """Format retrieved data for UI display."""
    context_str = "\n".join(f"\nContext {i + 1}\n{doc}" for i, doc in enumerate(docs))

    metadata_str = "\n".join(
        f"\nMetadata {i + 1}\n{meta}" for i, meta in enumerate(metas)
    )

    return context_str, metadata_str


def rag_query(
    question,
    top_k,
    course,
    lecture,
    professor,
    date,
    module,
    include_metadata,
):
    """Run full RAG pipeline."""

    if not question.strip():
        return "Please insert a question.", "", ""

    try:
        # Embed
        question_embedding = embed_question(question)

        # Filter
        metadata_filter = build_filter(course, lecture, professor, date)

        # Retrieval
        docs, metas = retrieve_documents(question_embedding, top_k, metadata_filter)

        # Prompt
        context_block = build_context_block(docs, metas, include_metadata)
        prompt = build_prompt(question, context_block)

        # LLM
        response = generate_response(
            llm=llm,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            config=GENERATION_CONFIG,
        )

        # UI formatting
        ctx, meta_str = format_retrieval_output(docs, metas)

        return response, ctx, meta_str

    except Exception as e:
        return f"Error: {e}", "", ""


def build_ui():
    """Create Gradio interface."""
    with gr.Blocks(title="Chat with Your AI Course") as demo:

        gr.Markdown("# 🎓 Chat with Your AI Course")
        gr.Markdown(
            "RAG system that answers questions about lectures from the Master's degree"
            "in AI at the University of Bologna.\n\n"
            "Developed as a project for the Big Data and Text Mining course."
        )

        question = gr.Textbox(
            label="Question",
            lines=3,
            placeholder="Ask something about your lectures...",
        )

        top_k = gr.Slider(1, 10, value=3, step=1, label="Top-K")

        gr.Markdown("## 🔎 Filters")

        with gr.Row():
            course = gr.Textbox(label="Course")
            module = gr.Textbox(label="Module")
            lecture = gr.Textbox(label="Lecture number")

        with gr.Row():
            professor = gr.Textbox(label="Professor")
            date = gr.Textbox(label="Date")

        include_metadata = gr.Checkbox(
            value=True,
            label="Include metadata in prompt",
        )

        submit = gr.Button("Search")

        answer = gr.Textbox(label="Answer")
        contexts = gr.Textbox(label="Retrieved contexts", lines=10)
        metadata_box = gr.Textbox(label="Retrieved metadata", lines=10)

        submit.click(
            rag_query,
            inputs=[
                question,
                top_k,
                course,
                lecture,
                professor,
                date,
                module,
                include_metadata,
            ],
            outputs=[answer, contexts, metadata_box],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(share=True)
