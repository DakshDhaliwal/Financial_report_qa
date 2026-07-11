import os
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="financial_docs")

def search(query, n=3):
    # turn your question into a number
    query_embedding = model.encode(query).tolist()

    # find the most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n
    )

    print(f"\n🔍 Query: {query}\n")
    for i, doc in enumerate(results["documents"][0]):
        page = results["metadatas"][0][i]["page"]
        type_ = results["metadatas"][0][i]["type"]
        print(f"Result {i+1} (page {page}, {type_}):")
        print(doc[:300])
        print("---")

# try these
search("what is the total revenue")
search("what are the risk factors")
search("operating expenses")