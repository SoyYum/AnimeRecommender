import os
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

df = pd.read_csv("data/cleaned_anime.csv")
synopsis = df["synopsis"][0]
print("Original Synopsis:\n")
print(synopsis)
print("\n" + "="*80 + "\n")
synopsis = synopsis.lower()
words = synopsis.split()
stop_words = set(stopwords.words("english"))

filtered = []
for word in words:
    if word not in stop_words:
        filtered.append(word)

cleaned = []
for word in filtered:
    cleaned.append(lemmatizer.lemmatize(word))

cleaned_synopsis = " ".join(cleaned)
print("Cleaned Synopsis:\n")
print(cleaned_synopsis)