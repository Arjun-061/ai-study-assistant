from sentence_transformers import SentenceTransformer
import numpy as np
import os

if not os.path.exists("data.txt"):
    study_content = """Python is a high-level programming language created by Guido van Rossum.
Lists in Python are mutable sequences that can hold multiple items.
Dictionaries store data using key-value pairs and are unordered.
NumPy is the fundamental package for scientific computing with Python.
Pandas is a powerful data analysis and manipulation library.
Machine learning models require training data and validation data.
Sentence Transformers create high-quality sentence embeddings.
Cosine similarity measures how similar two vectors are in direction.
RAG (Retrieval Augmented Generation) combines search with AI generation.
Vector databases efficiently store and query high-dimensional embeddings."""
    
    with open("data.txt", "w", encoding="utf-8") as f:
        f.write(study_content)
    print("✅ Created data.txt automatically!")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load study material
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Clean and split
documents = [line.strip() for line in text.split("\n") if line.strip()]

# Create embeddings
doc_embeddings = model.encode(documents)

# Simulated Endee storage
vector_db = list(zip(documents, doc_embeddings))

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Retrieve relevant chunks
def retrieve(query, top_k=3):
    query_embedding = model.encode([query])[0]
    
    scores = []
    for doc, emb in vector_db:
        score = cosine_similarity(query_embedding, emb)
        scores.append((doc, score))
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# Generate answer
def generate_answer(query):
    results = retrieve(query)
    context = " ".join([doc for doc, _ in results])
    return f"Based on your notes: {context}", results

# Run app
if __name__ == "__main__":
    print("📚 AI Study Assistant (RAG)")
    
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        
        answer, results = generate_answer(query)
        
        print("\n🧠 Answer:")
        print(answer)
        
        print("\n🔍 Top Matches:")
        for doc, score in results:
            print(f"{doc} (score: {score:.4f})")
