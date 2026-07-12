# app.py
import streamlit as st
import os
import fitz
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from query_router import classify_query


# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="Financial Report Q&A",
    page_icon="💹",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .source-badge {
        background: #1e3a5f;
        color: #7dd3fc;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin-top: 6px;
        display: inline-block;
    }
    .stat-box {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 8px;
    }
    .stat-number { font-size: 28px; font-weight: bold; color: #38bdf8; }
    .stat-label  { font-size: 12px; color: #94a3b8; margin-top: 4px; }
    .sample-btn  { margin: 3px 0; }
</style>
""", unsafe_allow_html=True)


# ── Load models once ──────────────────────────────────
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return embedding_model, llm

embedding_model, llm = load_models()

def get_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection("financial_docs")


# ── PDF Processing ────────────────────────────────────
def process_pdf(uploaded_file):
    with open("temp_report.pdf", "wb") as f:
        f.write(uploaded_file.read())

    text_chunks = []
    doc = fitz.open("temp_report.pdf")
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            words = text.split()
            for i in range(0, len(words), 500):
                chunk = " ".join(words[i:i+500])
                if chunk.strip():
                    text_chunks.append({
                        "text": chunk,
                        "page": page_num + 1,
                        "type": "text"
                    })

    table_chunks = []
    with pdfplumber.open("temp_report.pdf") as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                if table:
                    table_text = ""
                    for row in table:
                        if row:
                            clean_row = [str(cell) for cell in row if cell]
                            table_text += " | ".join(clean_row) + "\n"
                    if table_text.strip():
                        table_chunks.append({
                            "text": table_text,
                            "page": page_num + 1,
                            "type": "table"
                        })

    return text_chunks, table_chunks


def store_in_db(all_chunks, report_name):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    try:
        chroma_client.delete_collection("financial_docs")
    except:
        pass

    fresh_collection = chroma_client.get_or_create_collection("financial_docs")
    progress = st.progress(0)
    status = st.empty()

    for i, chunk in enumerate(all_chunks):
        embedding = embedding_model.encode(chunk["text"]).tolist()
        fresh_collection.add(
            ids=[f"{report_name}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "page": chunk["page"],
                "type": chunk["type"],
                "report": report_name
            }]
        )
        progress.progress((i + 1) / len(all_chunks))
        status.text(f"Processing chunk {i+1} of {len(all_chunks)}...")

    progress.empty()
    status.empty()
    return fresh_collection


# ── Answer Function ───────────────────────────────────
def answer_question(question):
    collection = get_collection()
    chunk_type = classify_query(question)
    query_embedding = embedding_model.encode(question).tolist()

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"type": chunk_type}
        )
    except Exception:
        # fallback: search all chunks if filtered search fails
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

    if not results["documents"][0]:
        return "I couldn't find relevant information in the report.", [], chunk_type

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        page = results["metadatas"][0][i]["page"]
        chunks.append({"text": doc, "page": page})

    context = ""
    for chunk in chunks:
        context += f"[Page {chunk['page']}]\n{chunk['text']}\n\n"

    prompt = f"""You are a professional financial analyst assistant.
Answer the question clearly and concisely using ONLY the context provided.
Always mention the specific page number(s) where you found the information.
Format numbers clearly (e.g. $416.2 billion, not 416161).
If the answer is not in the context, say "I couldn't find that in the report."

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Sorry, I encountered an error generating the answer: {str(e)}"

    pages = sorted(set([c["page"] for c in chunks]))
    return answer, pages, chunk_type


# ════════════════════════════════════════════════════════
# UI LAYOUT
# ════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💹 Financial Q&A")
    st.markdown("---")

    # upload section
    st.markdown("### 📂 Upload Report")
    uploaded_file = st.file_uploader(
        "Supports annual reports, 10-K, 10-Q filings",
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.success(f"📄 {uploaded_file.name}")
        file_size = round(uploaded_file.size / (1024*1024), 1)
        st.caption(f"Size: {file_size} MB")

        if st.button("⚙️ Process PDF", type="primary", use_container_width=True):
            with st.spinner("📖 Reading PDF..."):
                try:
                    text_chunks, table_chunks = process_pdf(uploaded_file)
                    all_chunks = text_chunks + table_chunks
                except Exception as e:
                    st.error(f"Failed to read PDF: {str(e)}")
                    st.stop()

            # show stats
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-number'>{len(text_chunks)}</div>
                    <div class='stat-label'>Text Chunks</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='stat-box'>
                    <div class='stat-number'>{len(table_chunks)}</div>
                    <div class='stat-label'>Tables Found</div>
                </div>""", unsafe_allow_html=True)

            with st.spinner("🧠 Building search index..."):
                try:
                    store_in_db(all_chunks, uploaded_file.name)
                    st.session_state.pdf_loaded = True
                    st.session_state.report_name = uploaded_file.name
                    st.session_state.messages = []  # clear old chat
                    st.success("✅ Ready! Ask your questions.")
                except Exception as e:
                    st.error(f"Failed to process: {str(e)}")
                    st.stop()

    # active report indicator
    if st.session_state.get("pdf_loaded"):
        st.markdown("---")
        st.markdown("### 📊 Active Report")
        st.info(f"📄 {st.session_state.get('report_name', '')}")

        # sample questions
        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What was the total revenue?",
            "What are the main risk factors?",
            "How did operating expenses change?",
            "What is the gross margin?",
            "What does management say about future outlook?",
            "What were the earnings per share?",
        ]

        for q in sample_questions:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state.pending_question = q

    st.markdown("---")
    st.caption("Built with ChromaDB · Groq · Streamlit")


# ── Main Area ─────────────────────────────────────────
if not st.session_state.get("pdf_loaded"):
    # welcome screen
    st.markdown("""
    ## Welcome to Financial Report Q&A 👋

    This tool lets you **chat with any financial PDF** — annual reports,
    10-K filings, 10-Q filings, and more.

    ### How it works:
    1. 📂 **Upload** your PDF from the sidebar
    2. ⚙️ **Process** it (takes ~30 seconds)
    3. 💬 **Ask** anything about the report

    ### What you can ask:
    - *"What was the total revenue this year?"*
    - *"What are the biggest risk factors?"*
    - *"How did gross margin change year over year?"*
    - *"What does management say about AI investments?"*
    """)

else:
    # chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "pages" in message:
                route = "📊 Tables" if message["chunk_type"] == "table" else "📝 Text"
                st.markdown(
                    f"<span class='source-badge'>📎 Pages {message['pages']}  ·  {route}</span>",
                    unsafe_allow_html=True
                )

    # handle sample question button clicks
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, pages, chunk_type = answer_question(question)
            st.write(answer)
            route = "📊 Tables" if chunk_type == "table" else "📝 Text"
            st.markdown(
                f"<span class='source-badge'>📎 Pages {pages}  ·  {route}</span>",
                unsafe_allow_html=True
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "pages": pages,
            "chunk_type": chunk_type
        })
        st.rerun()

    # chat input
    if question := st.chat_input("Ask anything about the report..."):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, pages, chunk_type = answer_question(question)
            st.write(answer)
            route = "📊 Tables" if chunk_type == "table" else "📝 Text"
            st.markdown(
                f"<span class='source-badge'>📎 Pages {pages}  ·  {route}</span>",
                unsafe_allow_html=True
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "pages": pages,
            "chunk_type": chunk_type
        })
        st.rerun()