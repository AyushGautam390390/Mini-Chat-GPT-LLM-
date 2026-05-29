from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import json
import faiss
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from tensorflow.keras.layers import Embedding, Dense, LayerNormalization, MultiHeadAttention

app = FastAPI(title="Mini LLM API")

# ── Config ──────────────────────────────────────────
SEQ_LEN   = 20
embed_dim  = 128
heads      = 4
ff_dim     = 256
max_len    = 100

# ── Custom Layers ────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_len, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.max_len    = max_len
        self.token_emb  = Embedding(vocab_size, embed_dim)
        self.pos_emb    = Embedding(max_len, embed_dim)

    def call(self, x):
        length    = tf.shape(x)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_emb(x) + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size, "embed_dim": self.embed_dim, "max_len": self.max_len})
        return config


@tf.keras.utils.register_keras_serializable()
class GPTBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.heads     = heads
        self.ff_dim    = ff_dim
        self.att       = MultiHeadAttention(num_heads=heads, key_dim=embed_dim)
        self.ffn       = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.norm1     = LayerNormalization()
        self.norm2     = LayerNormalization()

    def call(self, x):
        attn = self.att(x, x, use_causal_mask=True)
        x    = self.norm1(x + attn)
        x    = self.norm2(x + self.ffn(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "heads": self.heads, "ff_dim": self.ff_dim})
        return config


# ── Load Tokenizer ───────────────────────────────────
tokenizer  = Tokenizer.from_file("artifacts/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

# ── Load Model ───────────────────────────────────────
model = tf.keras.models.load_model(
    "artifacts/model.keras",
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "GPTBlock": GPTBlock
    },
    compile=False
)
print("✅ Model loaded")

# ── RAG ──────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index    = faiss.read_index("artifacts/faiss.index")

with open("artifacts/chunks.json") as f:
    chunks = json.load(f)

def retrieve(query, k=2):
    q_emb = embedder.encode([query])
    _, indices = index.search(np.array(q_emb), k)
    return [chunks[i]["text"] for i in indices[0]]

# ── Generate ─────────────────────────────────────────
def generate(seed, steps=100, temperature=0.7):
    ids = tokenizer.encode(seed).ids

    for _ in range(steps):
        seq = ids[-SEQ_LEN:]
        if len(seq) < SEQ_LEN:
            seq = [0] * (SEQ_LEN - len(seq)) + seq

        seq   = np.array(seq).reshape(1, SEQ_LEN)
        pred  = model.predict(seq, verbose=0)
        probs = pred[0, -1]
        probs = np.log(probs + 1e-9) / temperature
        probs = np.exp(probs)
        probs = probs / np.sum(probs)

        next_id = np.random.choice(len(probs), p=probs)
        ids.append(next_id)

    return tokenizer.decode(ids)

# ── Request Models ───────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 100
    temperature: float = 0.7

class RAGRequest(BaseModel):
    query: str
    steps: int = 100

# ── Routes ───────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Mini LLM API running 🚀"}

@app.post("/generate")
def generate_text(req: GenerateRequest):
    return {"prompt": req.prompt, "generated": generate(req.prompt, req.steps, req.temperature)}

@app.post("/rag-generate")
def rag_generate(req: RAGRequest):
    context = " ".join(retrieve(req.query))
    prompt  = " ".join((context + " " + req.query).split()[-SEQ_LEN:])
    return {"query": req.query, "generated": generate(prompt, req.steps)}

@app.post("/retrieve")
def retrieve_only(req: RAGRequest):
    return {"query": req.query, "chunks": retrieve(req.query)}

# ── Run ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)  