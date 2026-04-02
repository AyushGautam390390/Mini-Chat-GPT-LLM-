from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle

# --------------------------
# Load model + vocab
# --------------------------
model = tf.keras.models.load_model("gpt_model.h5")

with open("vocab.pkl", "rb") as f:
    word2id, id2word = pickle.load(f)

seq_len = 4  # keep same as training

# --------------------------
# RAG setup (load FAISS + embedder)
# --------------------------
from sentence_transformers import SentenceTransformer
import faiss

embedder = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("faiss.index")  # save this beforehand
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# --------------------------
# Request schema
# --------------------------
class Query(BaseModel):
    text: str

# --------------------------
# Retrieval
# --------------------------
def retrieve(query, k=2):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

# --------------------------
# Generate (with sampling)
# --------------------------
def generate(seed, steps=50, temperature=0.7):

    words = seed.lower().split()

    for _ in range(steps):

        seq = [word2id.get(w, 0) for w in words[-seq_len:]]

        if len(seq) < seq_len:
            seq = [0]*(seq_len-len(seq)) + seq

        seq = np.array(seq).reshape(1, seq_len)

        pred = model.predict(seq, verbose=0)

        probs = pred[0, -1]
        probs = np.log(probs + 1e-9) / temperature
        probs = np.exp(probs)
        probs = probs / np.sum(probs)

        next_id = np.random.choice(len(probs), p=probs)

        next_word = id2word[next_id]
        words.append(next_word)

    return " ".join(words)

# --------------------------
# RAG + Generate
# --------------------------
def generate_with_rag(query):
    context = retrieve(query)
    context_text = " ".join(context)

    prompt = context_text + " " + query

    # limit size (important)
    prompt_words = prompt.split()[-20:]
    prompt = " ".join(prompt_words)

    return generate(prompt)

# --------------------------
# API endpoints
# --------------------------
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Mini GPT + RAG API running"}

@app.post("/chat")
def chat(q: Query):
    response = generate_with_rag(q.text)
    return {"query": q.text, "response": response}