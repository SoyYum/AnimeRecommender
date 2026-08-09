import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/cleaned_anime.csv")
df = df.fillna("")

df["title_lower"] = df["title"].str.lower()

df["combined_features"] = (
    df["genres"] + " " + df["genres"] + " " + df["genres"] + " " +
    df["themes"] + " " +
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

titles = df["title_lower"].tolist()


# 🔹 Search
def search_anime(anime_name):
    anime_name = anime_name.lower()

    matches = df[df["title_lower"].str.contains(anime_name)]

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

def is_same_series(base, candidate):
    base_main = base.split(":")[0].lower()
    return base_main in candidate.lower()


# 🔹 Recommend from index
def recommend_from_index(index):
    scores = similarity[index]

    similarity_scores = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    base_title = df.iloc[index]["title"]

    for idx, score in similarity_scores[1:]:
        title = df.iloc[idx]["title"]

        if not is_same_series(base_title, title):
            recommendations.append(title)

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