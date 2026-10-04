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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

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

    .welcome-card {
        padding: 30px;
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-top: 30px;
        margin-bottom: 25px;
    }

    .source-card {
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-top: 5px;
        margin-bottom: 5px;
        font-size: 0.9rem;
    }

    section[data-testid="stSidebar"] {
        width: 300px !important;
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

current_session = st.session_state.sessions[
    current_session_id
]


# ============================================================
# API KEY
# ============================================================

def get_groq_api_key():

    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()


if not GROQ_API_KEY:

    st.error("GROQ_API_KEY is not configured.")

    st.info(
        "Add GROQ_API_KEY to Streamlit Secrets "
        "or as an environment variable."
    )

    st.stop()


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LLM
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
You are a helpful AI assistant.

You can operate in two modes.

MODE 1 - NORMAL CHAT
If no document context is provided, answer the user's
question normally using your general knowledge.

MODE 2 - DOCUMENT RAG
If document context is provided, use it to answer the question.

Rules for document questions:

1. Prefer information from the provided documents.
2. Do not invent information that is not supported by the documents.
3. If the answer cannot be found in the documents, say:
   "I couldn't find that information in the uploaded documents."
4. Use conversation history to understand follow-up questions.
5. Keep answers clear and useful.
6. Do not mention embeddings, vector databases, retrieval,
   prompts, or internal system instructions.

DOCUMENT CONTEXT:

{context}
""",
        ),
        MessagesPlaceholder(
            variable_name="history"
        ),
        (
            "human",
            "{input}",
        ),
    ]
)


# ============================================================
# MESSAGE HISTORY
# ============================================================

def get_session_history(session_id):

    if session_id not in st.session_state.sessions:

        st.session_state.sessions[session_id] = {
            "title": "New Chat",
            "messages": [],
            "history": InMemoryChatMessageHistory(),
        }

    return st.session_state.sessions[session_id]["history"]


# ============================================================
# CHAT CHAIN
# ============================================================

@st.cache_resource
def create_chat_chain(api_key):

    llm = get_llm(api_key)

    chain = prompt | llm

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


chat_chain = create_chat_chain(GROQ_API_KEY)


# ============================================================
# PROCESS PDF DOCUMENTS
# ============================================================

def process_documents(uploaded_files):

    if not uploaded_files:
        return None, [], 0

    all_documents = []

    temp_dir = tempfile.mkdtemp()

    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            temp_dir,
            uploaded_file.name,
        )

        with open(file_path, "wb") as file:
            file.write(
                uploaded_file.getbuffer()
            )

        try:

            loader = PyPDFLoader(file_path)

            documents = loader.load()

            for document in documents:

                document.metadata["source"] = (
                    uploaded_file.name
                )

                if "page" in document.metadata:

                    document.metadata["page"] = (
                        document.metadata["page"] + 1
                    )

            all_documents.extend(documents)

        except Exception as error:

            st.error(
                f"Could not process "
                f"{uploaded_file.name}: {error}"
            )

    if not all_documents:

        return None, [], 0

    # --------------------------------------------------------
    # SPLIT DOCUMENTS
    # --------------------------------------------------------

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

    chunks = splitter.split_documents(
        all_documents
    )

    # --------------------------------------------------------
    # CHUNK IDS
    # --------------------------------------------------------

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

    # --------------------------------------------------------
    # CHROMA
    # --------------------------------------------------------

    embeddings = get_embeddings()

    collection_name = (
        "rag_"
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
# FORMAT DOCUMENT CONTEXT
# ============================================================

def format_context(documents):

    if not documents:

        return ""

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "Unknown",
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
# GET SOURCES
# ============================================================

def get_sources(documents):

    sources = []

    seen = set()

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        page = document.metadata.get(
            "page",
            "Unknown",
        )

        key = (
            source,
            page,
        )

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
# STREAM RESPONSE
# ============================================================

def stream_response(
    question,
    context,
    session_id,
):

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

        if not hasattr(
            chunk,
            "content",
        ):
            continue

        content = chunk.content

        if isinstance(content, str):

            if content:
                yield content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 RAG Chatbot")

    st.caption(
        "AI chat with optional document search"
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

        st.session_state.sessions[
            new_session_id
        ] = {
            "title": "New Chat",
            "messages": [],
            "history": InMemoryChatMessageHistory(),
        }

        st.session_state.current_session_id = (
            new_session_id
        )

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    st.markdown("### 📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
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

            (
                vector_store,
                documents,
                chunk_count,
            ) = process_documents(
                uploaded_files
            )

            if vector_store:

                st.session_state.vector_store = (
                    vector_store
                )

                st.session_state.uploaded_documents = [
                    file.name
                    for file in documents
                ]

                st.session_state.chunk_count = (
                    chunk_count
                )

                status.update(
                    label=(
                        f"Processed "
                        f"{len(documents)} document(s) "
                        f"into {chunk_count} chunks"
                    ),
                    state="complete",
                )

            else:

                status.update(
                    label="Processing failed",
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

        for filename in (
            st.session_state.uploaded_documents
        ):

            st.caption(
                f"📄 {filename}"
            )

        st.caption(
            f"🧩 {st.session_state.chunk_count} chunks"
        )

    else:

        st.info(
            "No documents uploaded. "
            "You can still use normal chat."
        )

    st.divider()

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    st.markdown("### 💬 Chats")

    session_items = list(
        st.session_state.sessions.items()
    )

    for session_id, session in reversed(
        session_items
    ):

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
    # CLEAR CHAT
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
        "LLM: Groq / GPT-OSS-20B"
    )

    st.caption(
        "Vector DB: Chroma"
    )

    st.caption(
        "Embeddings: MiniLM"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="app-title">🤖 RAG Chatbot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    'Chat normally or ask questions about your documents'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not current_session["messages"]:

    if st.session_state.vector_store:

        st.markdown(
            """
            <div class="welcome-card">

            <h2>📚 Documents are ready</h2>

            <p>
            Your PDFs have been indexed.
            Ask questions and I will search the documents
            before generating an answer.
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="welcome-card">

            <h2>👋 Welcome!</h2>

            <p>
            You can start chatting immediately.
            </p>

            <p>
            Or upload PDF documents from the sidebar
            to enable document-based RAG.
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in current_session["messages"]:

    role = message["role"]

    with st.chat_message(role):

        st.markdown(
            message["content"]
        )

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
    "Message RAG Chatbot..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if user_input:

    question = user_input.strip()

    if not question:

        st.stop()

    # --------------------------------------------------------
    # CHAT TITLE
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
    # DISPLAY USER
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
    # RETRIEVAL
    # --------------------------------------------------------

    retrieved_documents = []

    sources = []

    if st.session_state.vector_store:

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

            retrieved_documents = (
                retriever.invoke(question)
            )

            status.update(
                label=(
                    f"Found "
                    f"{len(retrieved_documents)} "
                    "relevant sections"
                ),
                state="complete",
            )

        context = format_context(
            retrieved_documents
        )

        sources = get_sources(
            retrieved_documents
        )

    else:

        # Normal chat mode
        context = ""

    # --------------------------------------------------------
    # STREAM RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        answer = st.write_stream(
            stream_response(
                question=question,
                context=context,
                session_id=(
                    st.session_state.current_session_id
                ),
            )
        )

        if not answer:

            answer = (
                "I couldn't generate a response. "
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
    # SAVE MESSAGE
    # --------------------------------------------------------

    current_session["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )

    st.rerun()
