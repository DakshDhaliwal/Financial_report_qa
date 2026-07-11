import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2") 

# ── Connect to local ChromaDB ──────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="financial_docs")


# ── STEP A: Chop text into chunks ─────────────────────
def chunk_text(text_data, chunk_size=500):
    chunks = []

    for page in text_data:
        words = page["content"].split()
        
        # slide a window of chunk_size words across the page
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page["page"],
                    "type": "text"
                })

    print(f"✅ Created {len(chunks)} text chunks")
    return chunks


# ── STEP B: Convert tables to text chunks ─────────────
def chunk_tables(table_data):
    chunks = []

    for item in table_data:
        # turn each table row into a readable sentence
        rows = item["table"]
        table_text = ""
        for row in rows:
            if row:
                clean_row = [str(cell) for cell in row if cell]
                table_text += " | ".join(clean_row) + "\n"

        if table_text.strip():
            chunks.append({
                "text": table_text,
                "page": item["page"],
                "type": "table"
            })

    print(f"✅ Created {len(chunks)} table chunks")
    return chunks


# ── STEP C: Get embedding for one piece of text ───────
def get_embedding(text):
    return model.encode(text).tolist()


# ── STEP D: Store everything in ChromaDB ──────────────
def store_chunks(chunks):
    print("⏳ Generating embeddings and storing... (this takes a minute)")

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"])

        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "page": chunk["page"],
                "type": chunk["type"]
            }]
        )

        # show progress every 20 chunks
        if i % 20 == 0:
            print(f"  processed {i}/{len(chunks)} chunks...")

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")


# ── RUN IT ────────────────────────────────────────────
with open("text_output.json") as f:
    text_data = json.load(f)

with open("tables_output.json") as f:
    table_data = json.load(f)

text_chunks = chunk_text(text_data)
table_chunks = chunk_tables(table_data)

all_chunks = text_chunks + table_chunks
store_chunks(all_chunks)

print("\n🎉 Done! Your report is now searchable.")