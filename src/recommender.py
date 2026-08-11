import pandas as pd
from rapidfuzz import process, fuzz
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/cleaned_anime.csv")
df = df.fillna("")
embeddings = np.load("data/anime_embeddings.npy")

df["title"] = df["title"].str.lower().str.strip()
df["title_english"] = df["title_english"].fillna("").str.lower().str.strip()

df["combined_features"] = (
    (df["genres"] + " ") * 5 +
    (df["themes"] + " ") * 2 +
    df["demographics"] + " " +
    df["type"] + " " +
    df["source"] + " " +
    df["studios"] + " " +
    df["synopsis"]
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.8,
    min_df=2,
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
similarity = cosine_similarity(tfidf_matrix)

titles = (df["title"] + " " + df["title_english"]).tolist()

def search_anime(anime_name):
    anime_name = anime_name.lower().strip()

    matches = df[
        df["title"].str.contains(anime_name, na=False) |
        df["title_english"].str.contains(anime_name, na=False)
    ]

    matches = matches[
        matches["type"].isin(["TV", "Movie"])
    ]

    if not matches.empty:
        return "found", matches

    suggestions = process.extract(
        anime_name,
        titles,
        scorer=fuzz.QRatio,
        limit=4
    )

    suggestions = [s for s in suggestions if s[1] >= 70]

    if len(suggestions) == 0:
        return "not_found", None

    return "fuzzy", suggestions

def normalize_title(title):
    title = title.lower()

    remove_words = [
        "season",
        "2nd",
        "3rd",
        "4th",
        "5th",
        "final",
        "part",
        "movie",
        "ova",
        "ona"
    ]

    for word in remove_words:
        title = title.replace(word, "")

    title = re.sub(r'\d+', '', title)

    title = re.sub(r'[^a-z ]', '', title)

    return title.strip()


def is_same_series(base, candidate):

    base = normalize_title(base)
    candidate = normalize_title(candidate)

    return (
        base == candidate 
        or base in candidate 
        or candidate in base
    )

def explain_recommendation(base_idx, rec_idx):

    base = df.iloc[base_idx]
    rec = df.iloc[rec_idx]

    reasons = []

    if base["genres"] == rec["genres"]:
        reasons.append("Similar genres")

    if base["themes"] == rec["themes"]:
        reasons.append("Similar themes")

    if base["type"] == rec["type"]:
        reasons.append("Same format")

    return reasons

def recommend_from_index(index):
    scores = similarity[index]
    sbert_scores = cosine_similarity([embeddings[index]],embeddings)[0]
    final_scores = (
        0.4 * scores +
        0.6 * sbert_scores
    )
    similarity_scores = sorted(
        list(enumerate(final_scores)),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    base_title = df.iloc[index]["title"]

    for idx, score in similarity_scores:

        if idx == index:
            continue

        original_title = df.iloc[idx]["title"]

        if is_same_series(base_title, original_title):
            continue

        english_title = df.iloc[idx]["title_english"]

        title = english_title if english_title != "" else original_title

        recommendations.append((title,score))

        if len(recommendations) == 10:
            break
    return recommendations

def recommend(anime_name):
    status, data = search_anime(anime_name)

    if status == "not_found":
        return "not_found", None

    if status == "fuzzy":
        return "fuzzy", data

    matches = data

    if len(matches) > 1:
        return "multiple", matches

    index = matches.index[0]

    recommendations = recommend_from_index(index)

    return "success", recommendations