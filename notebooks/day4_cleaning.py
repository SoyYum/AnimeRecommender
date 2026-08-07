import pandas as pd
df = pd.read_csv("data/anime.csv")
columns = [
    "title",
    "synopsis",
    "genres",
    "themes",
    "demographics",
    "type",
    "source",
    "studios",
    "score",
    "episodes",
    "image_url"
]

df = df[columns]
df = df.dropna(subset=["synopsis"])

df = df.drop_duplicates()
df.to_csv("data/cleaned_anime.csv", index=False)
print(df.columns)