import pandas as pd
df = pd.read_csv("data/anime.csv")
columns = [
    "title",
    "title_english",
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
df["synopsis"] = df["synopsis"].fillna("")
df["synopsis"] = df["synopsis"].str.lower()
df["synopsis"] = df["synopsis"].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
df["synopsis"] = df["synopsis"].str.split().str[:80].str.join(" ")

df = df.drop_duplicates()
df.to_csv("data/cleaned_anime.csv", index=False)
print(df.columns)