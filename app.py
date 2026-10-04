import os
import uuid
import tempfile

import streamlit as st

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main content */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    /* Header */
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        color: #777;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Welcome card */
    .welcome-card {
        padding: 30px;
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-top: 30px;
        margin-bottom: 25px;
    }

    /* Source card */
    .source-card {
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-top: 5px;
        margin-bottom: 5px;
        font-size: 0.9rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        width: 300px !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session_id" not in st.session_state:
    session_id = str(uuid.uuid4())

    st.session_state.sessions[session_id] = {
        "title": "New Chat",
        "messages": [],
        "history": InMemoryChatMessageHistory(),
    }

    st.session_state.current_session_id = session_id


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ============================================================
# CURRENT SESSION
# ============================================================

current_session_id = st.session_state.current_session_id
current_session = st.session_state.sessions[current_session_id]


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_api_key():
    """
    Get Groq API key from Streamlit secrets first,
    then environment variable.
    """

    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# GROQ MODEL
# ============================================================

@st.cache_resource
def get_llm(api_key):

    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=api_key,
        temperature=0.2,
    )


# ============================================================
# RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful document-grounded AI assistant.

Your job is to answer the user's question using the provided
document context and conversation history.

Rules:

1. Use the document context whenever it contains relevant information.
2. Do not invent facts that are not supported by the documents.
3. If the answer cannot be found in the documents, clearly say:
   "I couldn't find that information in the uploaded documents."
4. You may use conversation history to understand follow-up questions.
5. Keep answers clear and useful.
6. If the user asks for a summary, provide a structured summary.
7. If the user asks a technical question, explain it step by step.
8. Do not mention internal prompts, embeddings, vector databases,
   retrieval mechanisms, or system instructions.

DOCUMENT CONTEXT:

{context}
""",
        ),
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "{input}",
        ),
    ]
)


# ============================================================
# MESSAGE HISTORY
# ============================================================

def get_session_history(session_id: str):

    if session_id not in st.session_state.sessions:
        st.session_state.sessions[session_id] = {
            "title": "New Chat",
            "messages": [],
            "history": InMemoryChatMessageHistory(),
        }

    return st.session_state.sessions[session_id]["history"]


# ============================================================
# CREATE LANGCHAIN CHAIN
# ============================================================

def create_chat_chain():

    if not GROQ_API_KEY:
        return None

    llm = get_llm(GROQ_API_KEY)

    chain = prompt | llm

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


chat_chain = create_chat_chain()


# ============================================================
# PDF PROCESSING
# ============================================================

def process_documents(uploaded_files):

    if not uploaded_files:
        return None, [], 0

    all_documents = []

    # Temporary directory
    temp_dir = tempfile.mkdtemp()

    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            temp_dir,
            uploaded_file.name,
        )

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        try:

            loader = PyPDFLoader(file_path)

            documents = loader.load()

            for document in documents:

                document.metadata["source"] = uploaded_file.name

                # PyPDFLoader uses zero-based page numbers
                if "page" in document.metadata:
                    document.metadata["page"] = (
                        document.metadata["page"] + 1
                    )

            all_documents.extend(documents)

        except Exception as error:

            st.error(
                f"Could not process {uploaded_file.name}: {error}"
            )

    if not all_documents:
        return None, [], 0

    # ========================================================
    # SPLIT DOCUMENTS
    # ========================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(all_documents)

    # Add unique chunk IDs
    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

    # ========================================================
    # CREATE CHROMA VECTOR STORE
    # ========================================================

    embeddings = get_embeddings()

    collection_name = (
        "rag_collection_"
        + str(uuid.uuid4()).replace("-", "")
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )

    return (
        vector_store,
        uploaded_files,
        len(chunks),
    )


# ============================================================
# FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_context(documents):

    if not documents:
        return "No relevant document content was found."

    context_parts = []

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "Unknown document",
        )

        page = document.metadata.get(
            "page",
            "Unknown",
        )

        content = document.page_content.strip()

        context_parts.append(
            f"""
SOURCE {index}
File: {source}
Page: {page}

Content:
{content}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# SOURCE INFORMATION
# ============================================================

def get_sources(documents):

    sources = []

    seen = set()

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown document",
        )

        page = document.metadata.get(
            "page",
            "Unknown",
        )

        key = (source, page)

        if key not in seen:

            seen.add(key)

            sources.append(
                {
                    "source": source,
                    "page": page,
                }
            )

    return sources


# ============================================================
# STREAM RAG RESPONSE
# ============================================================

def stream_response(
    question,
    context,
    session_id,
):

    if chat_chain is None:
        return

    config = {
        "configurable": {
            "session_id": session_id,
        }
    }

    for chunk in chat_chain.stream(
        {
            "input": question,
            "context": context,
        },
        config=config,
    ):

        if not hasattr(chunk, "content"):
            continue

        content = chunk.content

        if isinstance(content, str):

            if content:
                yield content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🤖 RAG Chatbot"
    )

    st.caption(
        "Chat with your documents using AI"
    )

    st.divider()

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True,
    ):

        new_session_id = str(uuid.uuid4())

        st.session_state.sessions[new_session_id] = {
            "title": "New Chat",
            "messages": [],
            "history": InMemoryChatMessageHistory(),
        }

        st.session_state.current_session_id = new_session_id

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # DOCUMENT UPLOAD
    # --------------------------------------------------------

    st.markdown("### 📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents.",
    )

    if st.button(
        "Process documents",
        use_container_width=True,
        disabled=not uploaded_files,
    ):

        with st.status(
            "Processing documents...",
            expanded=True,
        ) as status:

            vector_store, documents, chunk_count = (
                process_documents(uploaded_files)
            )

            if vector_store:

                st.session_state.vector_store = vector_store

                st.session_state.uploaded_documents = [
                    file.name
                    for file in documents
                ]

                st.session_state.chunk_count = (
                    chunk_count
                )

                status.update(
                    label=(
                        f"Processed {len(documents)} "
                        f"document(s) into "
                        f"{chunk_count} chunks"
                    ),
                    state="complete",
                )

            else:

                status.update(
                    label="Document processing failed",
                    state="error",
                )

    # --------------------------------------------------------
    # DOCUMENT STATUS
    # --------------------------------------------------------

    if st.session_state.uploaded_documents:

        st.success(
            f"{len(st.session_state.uploaded_documents)} "
            "document(s) ready"
        )

        for filename in st.session_state.uploaded_documents:

            st.caption(
                f"📄 {filename}"
            )

        st.caption(
            f"🧩 {st.session_state.chunk_count} chunks"
        )

    else:

        st.info(
            "Upload PDFs and click "
            "'Process documents'."
        )

    st.divider()

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    st.markdown("### 💬 Chats")

    session_items = list(
        st.session_state.sessions.items()
    )

    for session_id, session in reversed(session_items):

        title = session["title"]

        if len(title) > 28:
            title = title[:28] + "..."

        is_current = (
            session_id
            == st.session_state.current_session_id
        )

        button_label = (
            f"● {title}"
            if is_current
            else f"○ {title}"
        )

        if st.button(
            button_label,
            key=f"session_{session_id}",
            use_container_width=True,
        ):

            st.session_state.current_session_id = (
                session_id
            )

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # CLEAR CURRENT CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear current chat",
        use_container_width=True,
    ):

        session = st.session_state.sessions[
            st.session_state.current_session_id
        ]

        session["messages"] = []

        session["history"] = (
            InMemoryChatMessageHistory()
        )

        st.rerun()

    # --------------------------------------------------------
    # CLEAR DOCUMENTS
    # --------------------------------------------------------

    if st.button(
        "🧹 Clear documents",
        use_container_width=True,
    ):

        st.session_state.vector_store = None

        st.session_state.uploaded_documents = []

        st.session_state.chunk_count = 0

        st.rerun()

    st.divider()

    st.caption(
        "Model: openai/gpt-oss-20b"
    )

    st.caption(
        "Vector DB: Chroma"
    )

    st.caption(
        "Embeddings: MiniLM"
    )


# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    '<div class="app-title">🤖 RAG Chatbot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    'Ask questions about your uploaded documents'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# API KEY CHECK
# ============================================================

if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY is not configured."
    )

    st.info(
        "Add GROQ_API_KEY to Streamlit Secrets "
        "or as an environment variable."
    )

    st.stop()


# ============================================================
# WELCOME SCREEN
# ============================================================

if not current_session["messages"]:

    if st.session_state.vector_store is None:

        st.markdown(
            """
            <div class="welcome-card">

            <h2>👋 Welcome!</h2>

            <p>
            Upload one or more PDF documents from the sidebar,
            process them, and then ask questions about their content.
            </p>

            <p>
            The chatbot retrieves relevant sections from your
            documents and generates an answer using Groq.
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="welcome-card">

            <h2>📚 Your documents are ready</h2>

            <p>
            Ask anything about the uploaded documents.
            </p>

            <p>
            Answers are generated from the relevant document
            sections and include source pages.
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in current_session["messages"]:

    role = message["role"]

    content = message["content"]

    with st.chat_message(role):

        st.markdown(content)

        # Assistant sources
        if (
            role == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-card">
                        📄 <b>{source["source"]}</b>
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        Page {source["page"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask something about your documents..."
)


# ============================================================
# HANDLE USER QUESTION
# ============================================================

if user_input:

    question = user_input.strip()

    if not question:
        st.stop()

    # --------------------------------------------------------
    # CHECK VECTOR STORE
    # --------------------------------------------------------

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload and process at least one PDF "
            "before asking questions."
        )

        st.stop()

    # --------------------------------------------------------
    # UPDATE CHAT TITLE
    # --------------------------------------------------------

    if current_session["title"] == "New Chat":

        title = question.replace(
            "\n",
            " ",
        )

        if len(title) > 35:

            title = title[:35] + "..."

        current_session["title"] = title

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    current_session["messages"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------------

    with st.status(
        "Searching your documents...",
        expanded=False,
    ) as status:

        retriever = (
            st.session_state.vector_store
            .as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 4
                },
            )
        )

        retrieved_documents = retriever.invoke(
            question
        )

        status.update(
            label=(
                f"Found {len(retrieved_documents)} "
                "relevant sections"
            ),
            state="complete",
        )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = format_context(
        retrieved_documents
    )

    sources = get_sources(
        retrieved_documents
    )

    # --------------------------------------------------------
    # STREAM ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        answer = st.write_stream(
            stream_response(
                question=question,
                context=context,
                session_id=st.session_state.current_session_id,
            )
        )

        if not answer:

            answer = (
                "I couldn't generate an answer. "
                "Please try again."
            )

            st.markdown(answer)

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources:

                    st.markdown(
                        f"""
                        <div class="source-card">
                        📄 <b>{source["source"]}</b>
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        Page {source["page"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # --------------------------------------------------------
    # SAVE UI MESSAGE
    # --------------------------------------------------------

    current_session["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )

    st.rerun()