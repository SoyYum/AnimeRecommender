import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/cleaned_anime.csv")
df = df.fillna("")

df["combined_features"] = (
    df["genres"] + " " +
    df["themes"] + " " +
    df["demographics"] + " " +
    df["type"] + " " +
    df["source"] + " " +
    df["studios"] + " " +
    df["synopsis"]
)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

similarity = cosine_similarity(tfidf_matrix)
titles = df["title"].str.lower().tolist()
def search_anime(anime_name):

    matches = df[df["title"].str.lower().str.contains(anime_name.lower())]

    if not matches.empty:
        return "found", matches

    suggestions = process.extract(
        anime_name.lower(),
        titles,
        scorer=fuzz.QRatio,
        limit=4
    )

    suggestions = [
    s for s in suggestions
    if s[1] >= 70
]

    if len(suggestions) == 0:
        return "not_found", None

    return "fuzzy", suggestions
def recommend_from_index(index):

    scores = similarity[index]

    similarity_scores = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for idx, score in similarity_scores[1:11]:
        recommendations.append(df.iloc[idx]["title"])

    return recommendations
def recommend(anime_name):

    status, data = search_anime(anime_name)

    if status == "fuzzy":
        return "fuzzy", data

    matches = data

    if len(matches) > 1:
        return "multiple", matches

    index = matches.index[0]

    recommendations = recommend_from_index(index)

    return "success", recommendations

anime_name = input("Enter an anime name: ")

status, data = recommend(anime_name)

if status == "not_found":
    print("Anime not found!")
elif status == "fuzzy":

    print("\nAnime not found!")

    print("\nDid you mean:")

    for i, suggestion in enumerate(data, start=1):
        print(f"{i}. {suggestion[0]} ({suggestion[1]:.1f}%)")
elif status == "multiple":

    print("\nMultiple matches found:\n")

    for i, title in enumerate(data["title"], start=1):
        print(f"{i}. {title}")

    choice = int(input("\nChoose an anime: "))

    if 1 <= choice <= len(data):
        selected_index = data.index[choice - 1]

        recommendations = recommend_from_index(selected_index)

        print("\nTop Recommendations:\n")

        for i, anime in enumerate(recommendations, start=1):
            print(f"{i}. {anime}")

    else:
        print("Invalid choice!")

elif status == "success":

    print("\nTop Recommendations:\n")

    for i, anime in enumerate(data, start=1):
        print(f"{i}. {anime}")