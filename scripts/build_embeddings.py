import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


df = pd.read_csv("data/cleaned_anime.csv")
df = df.fillna("")


model = SentenceTransformer("all-MiniLM-L6-v2")

df["combined_features"] = (
    (df["genres"] + " ") * 5 +
    (df["themes"] + " ") * 2 +
    df["demographics"] + " " +
    df["type"] + " " +
    df["source"] + " " +
    df["studios"] + " " +
    df["synopsis"]
)

embeddings = model.encode(
    df["combined_features"].tolist(),
    batch_size=32,
    show_progress_bar=True
)


np.save(
    "data/anime_embeddings.npy",
    embeddings
)


print("Embeddings saved!")
print("Shape:", embeddings.shape)