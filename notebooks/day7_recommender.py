import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/cleaned_anime.csv")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["synopsis"])

similarity = cosine_similarity(tfidf_matrix)
def recommend(anime_name):
    matches = df[df["title"].str.lower().str.contains(anime_name.lower())]

    if matches.empty:
        return "not_found", None
    if len(matches) > 1:
        return "multiple", matches
    index = matches.index[0]

    scores = similarity[index]

    similarity_scores = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[1:11]:
        recommendations.append(df.iloc[index]["title"])

    return "success", recommendations

anime_name = input("Enter an anime name: ")

status, data = recommend(anime_name)

if status == "not_found":
    print("Anime not found!")

elif status == "multiple":

    print("\nMultiple matches found:\n")

    for i, title in enumerate(data["title"], start=1):
        print(f"{i}. {title}")

    choice = int(input("\nChoose an anime: "))

    if 1 <= choice <= len(data):
        selected_anime = data.iloc[choice - 1]["title"]

        status, recommendations = recommend(selected_anime)

        print("\nTop Recommendations:\n")

        for i, anime in enumerate(recommendations, start=1):
            print(f"{i}. {anime}")

    else:
        print("Invalid choice!")

elif status == "success":

    print("\nTop Recommendations:\n")

    for i, anime in enumerate(data, start=1):
        print(f"{i}. {anime}")