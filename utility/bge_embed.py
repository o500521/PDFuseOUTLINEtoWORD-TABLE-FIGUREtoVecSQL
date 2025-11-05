from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

def embed_text(text: str):
    return model.encode(text, normalize_embeddings=True).tolist()