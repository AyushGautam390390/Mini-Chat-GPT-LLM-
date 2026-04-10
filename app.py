# here is the upgraded version of backend deployment
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import faiss
from sentence_transformers import SentenceTransformer

# ✅ import memory functions (FROM YOUR db.py)
from db import save_message, load_memory

app = FastAPI(title="Mini LLM API")

# ── Config ──────────────────────────────────────────
SEQ_LEN   = 20
vocab_size = 4952
embed_dim  = 128
heads      = 4
ff_dim     = 256
max_len    = 100

# ── Custom Layers ────────────────────────────────────
from tensorflow.keras.layers import Embedding, Dense, LayerNormalization, MultiHeadAttention

@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_len, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb   = Embedding(max_len, embed_dim)

    def call(self, x):
        length    = tf.shape(x)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        positions = self.pos_emb(positions)
        x         = self.token_emb(x)
        return x + positions


@tf.keras.utils.register_keras_serializable()
class GPTBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.att   = MultiHeadAttention(num_heads=heads, key_dim=embed_dim)
        self.ffn   = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

    def call(self, x):
        attn = self.att(x, x, use_causal_mask=True)
        x    = self.norm1(x + attn)
        ffn  = self.ffn(x)
        x    = self.norm2(x + ffn)
        return x


# ── Load Model ───────────────────────────────────────
with open("artifacts/vocab.json") as f:
    vocab_data = json.load(f)

word2id = vocab_data["word2id"]
id2word = {int(k): v for k, v in vocab_data["id2word"].items()}

model = tf.keras.models.load_model(
    "artifacts/model.keras",
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "GPTBlock": GPTBlock
    },
    compile=False
)

# ── RAG Setup ────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index    = faiss.read_index("artifacts/faiss.index")

with open("artifacts/chunks.json") as f:
    chunks = json.load(f)

def retrieve(query, k=2):
    q_emb = embedder.encode([query])
    _, indices = index.search(np.array(q_emb), k)
    return [chunks[i] for i in indices[0]]


# ── Text Generation ──────────────────────────────────
def generate(seed, steps=100, temperature=0.7):
    words = seed.lower().split()

    for _ in range(steps):
        seq = [word2id.get(w, 0) for w in words[-SEQ_LEN:]]

        if len(seq) < SEQ_LEN:
            seq = [0] * (SEQ_LEN - len(seq)) + seq

        seq   = np.array(seq).reshape(1, SEQ_LEN)
        pred  = model.predict(seq, verbose=0)

        probs = pred[0, -1]
        probs = np.log(probs + 1e-9) / temperature
        probs = np.exp(probs)
        probs = probs / np.sum(probs)

        next_id = np.random.choice(len(probs), p=probs)
        words.append(id2word[next_id])

    return " ".join(words)


# ── Request Models ───────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 100
    temperature: float = 0.7


class RAGRequest(BaseModel):
    user_id: str
    session_id: str
    query: str
    steps: int = 100


# ── Routes ───────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Mini LLM API running 🚀"}


@app.post("/generate")
def generate_text(req: GenerateRequest):
    return {
        "prompt": req.prompt,
        "generated": generate(req.prompt, req.steps, req.temperature)
    }


@app.post("/rag-generate")
def rag_generate(req: RAGRequest):

    # 🔹 1. Load memory
    memory = load_memory(req.user_id, req.session_id)

    memory_text = ""
    for role, msg in memory:
        memory_text += f"{role}: {msg} "

    # 🔹 2. Retrieve RAG context
    retrieved_docs = retrieve(req.query)
    context = " ".join(retrieved_docs)

    # 🔹 3. Combine memory + context + query
    full_input = f"{memory_text} {context} {req.query}"

    # 🔹 4. Trim to model limit
    prompt = " ".join(full_input.split()[-SEQ_LEN:])

    # 🔹 5. Generate response
    response = generate(prompt, req.steps)

    # 🔹 6. Save chat
    save_message(req.user_id, req.session_id, "user", req.query)
    save_message(req.user_id, req.session_id, "bot", response)

    return {
        "query": req.query,
        "generated": response
    }


@app.post("/retrieve")
def retrieve_only(req: RAGRequest):
    return {
        "query": req.query,
        "chunks": retrieve(req.query)
    }

@app.get("/chats")
def get_chats():
    from db import cursor
    cursor.execute("SELECT * FROM chats")
    return {"data": cursor.fetchall()}

@app.get("/check-db")
def check_db():
    from db import cursor
    cursor.execute("SELECT * FROM chats")
    return {"data": cursor.fetchall()}
# ── Run ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
