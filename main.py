import sys
from src.recommender import recommend, recommend_from_index, recommend_from_query

if len(sys.argv) >= 2:
    user_input = " ".join(sys.argv[1:])
else:
    user_input = input("Enter an anime name or description: ")

status, data = recommend(user_input)

if status == "not_found":
    recommendations = recommend_from_query(user_input)

    print("\nTop Recommendations:\n")

    for i, (anime, score, reasons) in enumerate(recommendations, start=1):
        print(f"{i}. {anime}")
        print(f"   Similarity: {score:.2f}")
        print(f"   Why: {' | '.join(reasons)}")

elif status == "fuzzy":
    print("\nAnime not found!")
    print("\nDid you mean:")

    for i, suggestion in enumerate(data, start=1):
        print(f"{i}. {suggestion[0]} ({suggestion[1]:.1f}%)")

elif status == "multiple":
    print("\nMultiple matches found:\n")

    for i, idx in enumerate(data.index, start=1):
        row = data.loc[idx]
        title = row["title_english"] if row["title_english"] != "" else row["title"]
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

        for i, (anime, score, reasons) in enumerate(recommendations, start=1):
            print(f"{i}. {anime}")
            print(f"   Similarity: {score:.2f}")
            print(f"   Why: {' | '.join(reasons)}")
    else:
        print("Invalid choice!")

elif status == "success":
    print("\nTop Recommendations:\n")

    for i, (anime, score, reasons) in enumerate(data, start=1):
        print(f"{i}. {anime}")
        print(f"   Similarity: {score:.2f}")
        print(f"   Why: {' | '.join(reasons)}")