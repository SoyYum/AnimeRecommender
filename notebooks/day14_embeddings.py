from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

text1 = "A student discovers a mysterious time machine."

text2 = "A young boy finds a device that allows him to travel through time."

text3 = "A group of warriors fight monsters to protect their kingdom."

embedding1 = model.encode(text1)
embedding2 = model.encode(text2)
embedding3 = model.encode(text3)

similarity_12 = cosine_similarity(
    [embedding1],
    [embedding2]
)[0][0]

similarity_13 = cosine_similarity(
    [embedding1],
    [embedding3]
)[0][0]

print("Similarity between text 1 and text 2:", similarity_12)
print("Similarity between text 1 and text 3:", similarity_13)