import pandas as pd
df = pd.read_csv("data/anime.csv")
columns = [
    "title",
    "synopsis",
    "genres",
    "score",
    "episodes",
    "type",
    "image_url"
]

df = df[columns]
df = df.dropna(subset=["synopsis"])
duplicates = df[df["title"].duplicated(keep=False)]

df = df.drop_duplicates()
df.to_csv("data/cleaned_anime.csv", index=False)