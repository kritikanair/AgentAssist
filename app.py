
import streamlit as st

from rag import (
    load_embedding_model,
    create_document_records,
    index_documents,
)
from agent import run_agent
from tools import create_report_markdown, create_report_filename


def clear_document_workspace():
    """Clear the active document index and related session state."""
    client = st.session_state.get("chroma_client")
    collection = st.session_state.get("collection")

    if collection is not None:
        existing = collection.get()
        existing_ids = existing.get("ids", [])

        for start in range(0, len(existing_ids), 500):
            collection.delete(
                ids=existing_ids[start:start + 500]
            )

    # Clear the active document references.
    st.session_state.collection = None
    st.session_state.chroma_client = None
    st.session_state.indexed_files = []
    st.session_state.messages = []



st.set_page_config(
    page_title="AgentAssist",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AgentAssist")
st.caption("AI-Powered Document Research and Task Automation")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if st.session_state.embedding_model is None:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedding_model = load_embedding_model()

if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = None

if "collection" not in st.session_state:
    st.session_state.collection = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

with st.sidebar:
    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

if st.sidebar.button("Clear Documents"):
    try:
        clear_document_workspace()
        st.sidebar.success("Document workspace cleared.")
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"Could not clear documents: {exc}")
    
if st.button("Index Documents", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF or TXT file.")
    else:
        try:
            with st.spinner("Extracting text and indexing documents..."):
                records = create_document_records(uploaded_files)

                if not records:
                    st.warning(
                        "No readable text was found. "
                        "Try another PDF or TXT file."
                    )
                else:
                    client, collection = index_documents(
                        records,
                        st.session_state.embedding_model,
                    )

                    st.session_state.chroma_client = client
                    st.session_state.collection = collection
                    st.session_state.indexed_files = sorted(
                        {record["source"] for record in records}
                    )

                    # Clear previous conversation so answers from
                    # earlier documents are not shown as current.
                    st.session_state.messages = []

                    st.success(
                        f"Indexed {len(records)} chunks from "
                        f"{len(st.session_state.indexed_files)} file(s)."
                    )

        except Exception as exc:
            st.error(f"Document indexing failed: {exc}")


    if st.session_state.collection is not None:
        st.success("Knowledge base ready")
        st.caption(
            f"Files: {len(st.session_state.indexed_files)} · "
            f"Chunks: {st.session_state.collection.count()}"
        )

    st.divider()
    st.markdown("**Agent capabilities**")
    st.markdown(
        """
        - Document question answering
        - Calculator
        - Structured report generation
        - General AI chat
        """
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.subheader("Chat with your AI agent")

if st.session_state.collection is None:
    st.info(
        "You can ask general questions now. "
        "Upload and index a document to enable document search and reports."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("trace"):
            with st.expander("Agent execution trace"):
                for item in message["trace"]:
                    st.write(f"✓ {item}")

        if message.get("sources"):
            with st.expander("Retrieved document sources"):
                for source in message["sources"]:
                    page = source.get("page")
                    page_text = f", page {page}" if page else ""
                    st.markdown(f"**{source['source']}{page_text}**")
                    st.caption(source["text"])


prompt = st.chat_input("Ask a question or request a task...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is processing your request..."):
            try:
                result = run_agent(
                    question=prompt,
                    collection=st.session_state.collection,
                    embedding_model=st.session_state.embedding_model,
                )

                answer = result["answer"]
                st.markdown(answer)

                with st.expander("Agent execution trace", expanded=True):
                    st.write(f"**Selected route:** {result['route']}")

                    for item in result["trace"]:
                        st.write(f"✓ {item}")

                if result["sources"]:
                    with st.expander("Retrieved document sources"):
                        for source in result["sources"]:
                            page = source.get("page")
                            page_text = f", page {page}" if page else ""
                            st.markdown(
                                f"**{source['source']}{page_text}**"
                            )
                            st.caption(source["text"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "trace": result["trace"],
                        "sources": result["sources"],
                    }
                )

            except Exception as exc:
                st.error("The agent workflow failed.")
                st.exception(exc)
                st.session_state.messages.pop()
