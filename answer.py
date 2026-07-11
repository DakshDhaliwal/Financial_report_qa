# answer.py

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq  # swap this if using Gemini
from query_router import classify_query

load_dotenv()

# ── Load models & DB ──────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="financial_docs")
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── STEP A: Retrieve relevant chunks ─────────────────
def retrieve(question, chunk_type=None, n=5):
    # turn question into numbers
    query_embedding = model.encode(question).tolist()

    # filter by type if specified
    where_filter = {"type": chunk_type} if chunk_type else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where=where_filter
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        page = results["metadatas"][0][i]["page"]
        chunks.append({
            "text": doc,
            "page": page
        })

    return chunks


# ── STEP B: Ask LLM to answer using chunks ────────────
def ask_llm(question, chunks):
    # build context from chunks
    context = ""
    for chunk in chunks:
        context += f"[Page {chunk['page']}]\n{chunk['text']}\n\n"

    prompt = f"""You are a financial analyst assistant.
Answer the question using ONLY the context provided below.
Always mention the page number where you found the answer.
If the answer is not in the context, say "I couldn't find that in the report."

Context:
{context}

Question: {question}

Answer:"""

    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# ── STEP C: Full pipeline ─────────────────────────────
def answer_question(question):
    print(f"\n🔍 Question: {question}")

    # route the question
    chunk_type = classify_query(question)
    print(f"📌 Routing to: {chunk_type} chunks")

    # retrieve relevant chunks
    chunks = retrieve(question, chunk_type=chunk_type)
    print(f"📄 Found {len(chunks)} relevant chunks")

    # get answer from LLM
    answer = ask_llm(question, chunks)

    # show sources
    pages = list(set([c["page"] for c in chunks]))
    pages.sort()

    print(f"\n💬 Answer:\n{answer}")
    print(f"\n📎 Sources: Pages {pages}")
    print("\n" + "="*60)

    return answer, pages


# ── TEST IT ───────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What was the total revenue?",
        "What are the main risk factors?",
        "How much were the operating expenses?",
    ]

    for q in questions:
        answer_question(q)