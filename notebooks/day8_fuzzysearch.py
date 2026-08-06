from rapidfuzz import process
from rapidfuzz import fuzz
import pandas as pd
df = pd.read_csv("data/cleaned_anime.csv")
titles = df["title"].str.lower().tolist()
anime_name = input("Enter anime name: ").lower()

print(process.extract(anime_name, titles, scorer=fuzz.QRatio))