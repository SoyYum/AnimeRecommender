import sys
from src.recommender import recommend, recommend_from_index

if len(sys.argv) >= 2:
    anime_name = " ".join(sys.argv[1:])
else:
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

    try:
        choice = int(input("\nChoose an anime: "))
    except ValueError:
        print("Invalid input!")
        exit()

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