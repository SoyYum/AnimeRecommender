import pandas as pd
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/cleaned_anime.csv")
df = df.fillna("")

df["combined_features"] = (
    (df["genres"] + " ") * 5 +
    (df["themes"] + " ") * 2 +
    df["demographics"] + " " +
    df["type"] + " " +
    df["source"] + " " +
    df["studios"] + " " +
    df["synopsis"]
)

custom_stopwords = [
    "life", "story", "world", "people", "man", "woman",
    "boy", "girl", "day", "way", "new", "one",
    "school", "student", "friend", "friends"
]

all_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords))

vectorizer = TfidfVectorizer(
    stop_words=all_stopwords,
    ngram_range=(1, 2),
    max_df=0.8,
    min_df=2,
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

similarity = cosine_similarity(tfidf_matrix)

df.to_csv("data/processed_anime.csv", index=False)

with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

print("✅ Model built and saved successfully!")