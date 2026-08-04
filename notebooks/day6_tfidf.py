import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv("data/cleaned_anime.csv")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["synopsis"])
similarity = cosine_similarity(tfidf_matrix)
anime_name = "Made in Abyss"
index = df[df["title"] == anime_name].index[0]
scores = similarity[index]
similarity_scores = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1],
    reverse=True
)
for anime in similarity_scores[1:11]:
    print(df.iloc[anime[0]]["title"], anime[1])
#print(list(vectorizer.vocabulary_.items())[:20])
#print(tfidf_matrix[0])