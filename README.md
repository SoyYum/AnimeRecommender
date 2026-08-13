# 🎌 WeebWise

## Anime Recommendation System

WeebWise is a content-based anime recommendation system that helps users discover anime based on an anime title or a description of what they want to watch.

Instead of relying only on exact keyword matching, WeebWise analyzes anime synopses and metadata to find anime with similar content and characteristics.

---

## ✨ Features

- 🔎 Search using an anime title or natural-language description
- 🧠 Content-based and semantic anime recommendations
- 🎯 Ranked recommendations based on similarity
- 🖼️ Anime posters displayed with recommendations
- ⭐ Similarity score for every recommendation
- 📺 Anime type and episode count
- ⏱️ Episode duration
- 🎭 Genres and themes
- 🏢 Studio information
- 📚 Source material
- 🚫 Filters extremely low-popularity anime
- 🔀 Uses Maximal Marginal Relevance (MMR) to improve recommendation diversity
- 🎨 Interactive Streamlit interface

---

## 🧠 How It Works

WeebWise combines traditional NLP techniques with semantic embeddings to generate recommendations.

### 1. Data Processing

The anime dataset is cleaned and prepared before being used by the recommendation system.

The system uses information such as:

- Anime title
- English title
- Synopsis
- Genres
- Themes
- Demographics
- Type
- Source
- Studios
- Score
- Episodes
- Image URL

The synopsis is normalized and processed before being used for recommendation.

### 2. TF-IDF

TF-IDF is used to represent important words in anime descriptions.

This allows the system to identify anime that share meaningful terms and content.

### 3. Sentence-BERT

Sentence-BERT embeddings are used to capture semantic similarities between anime descriptions.

This allows anime to be considered similar even when their descriptions do not use exactly the same words.

### 4. Similarity Search

When a user enters an anime title or description, WeebWise processes the query and compares it against the anime dataset.

Cosine similarity is used to measure the similarity between the query and candidate anime.

Higher similarity scores indicate stronger matches.

### 5. Popularity Filtering

Extremely low-popularity anime can sometimes introduce noisy recommendations.

WeebWise therefore removes anime below the chosen popularity threshold before generating the final recommendation list.

This helps keep recommendations more useful while still allowing less mainstream anime to appear when they are relevant.

### 6. MMR Recommendation Ranking

Maximal Marginal Relevance (MMR) is used to improve the diversity of the final recommendations.

Without diversity control, a query could produce many recommendations that are extremely similar to each other.

MMR balances:

- Relevance to the user's query
- Diversity between recommended anime

This allows WeebWise to provide a more varied recommendation list.

---

## 🔄 Recommendation Pipeline

    User Query
         │
         ▼
    Query Processing
         │
         ▼
    Feature Representation
         │
         ├───────────────┐
         ▼               ▼
       TF-IDF      Sentence-BERT
         │               │
         └───────┬───────┘
                 ▼
         Similarity Search
                 │
                 ▼
        Candidate Retrieval
                 │
                 ▼
        Popularity Filtering
                 │
                 ▼
                MMR
                 │
                 ▼
       Ranked Recommendations
                 │
                 ▼
           Streamlit UI

---

## 🖥️ User Interface

The application provides a simple search interface where users can enter either an anime title or describe the kind of anime they want.

### Example

    Jujutsu Kaisen

or:

    I want a dark fantasy anime with supernatural powers,
    intense fights and strong characters.

WeebWise then returns a ranked list of relevant anime.

Each recommendation provides information such as:

- Anime title
- Similarity score
- Type
- Number of episodes
- Episode duration
- Genres
- Themes
- Studio
- Source
- Anime poster

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning & NLP

- Scikit-learn
- TF-IDF
- Sentence-BERT
- Cosine Similarity
- Maximal Marginal Relevance (MMR)

### Application

- Streamlit
- HTML/CSS

### Development Tools

- Git
- GitHub
- VS Code

---

## 📁 Project Structure

    AnimeRecommender/
    │
    ├── data/
    │   ├── anime.csv
    │   ├── cleaned_anime.csv
    │   ├── processed_anime.csv
    │   └── anime_embeddings.npy
    │
    ├── notebooks/
    │   ├── day3_pandas.py
    │   ├── day4_cleaning.py
    │   ├── day5_nlp.py
    │   ├── day6_tfidf.py
    │   ├── day7_recommender.py
    │   ├── day8_fuzzysearch.py
    │   ├── day9_featureengineering.py
    │   └── day14_embeddings.py
    │
    ├── scripts/
    │   ├── build_embeddings.py
    │   └── build_model.py
    │
    ├── src/
    │   └── recommender.py
    │
    ├── app.py
    ├── main.py
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

## ⚙️ Installation

Clone the repository:

    git clone https://github.com/SoyUm/AnimeRecommender.git

Enter the project directory:

    cd AnimeRecommender

Create a virtual environment:

    python -m venv .venv

Activate the environment on Windows:

    .venv\Scripts\activate

Install the required dependencies:

    pip install -r requirements.txt

---

## ▶️ Run Locally

Start the Streamlit application:

    streamlit run app.py

The application will normally be available at:

    http://localhost:8501

---

## 🔍 Example Queries

WeebWise accepts both anime titles and natural-language queries.

### Anime Titles

    Dragon Ball Z

    JoJo's Bizarre Adventure

    Jujutsu Kaisen

### Natural-Language Queries

    I want an anime with intense battles and supernatural powers.

    I want a dark fantasy anime with powerful characters and a serious story.

    I want something similar to a long-running shonen anime with lots of action.

---

## 🎯 Project Objective

The goal of WeebWise is to explore how machine learning and natural-language processing can be used to build a practical recommendation system.

The project combines traditional text-processing techniques such as TF-IDF with modern semantic embeddings to improve the quality of content-based anime recommendations.

---

## 🚀 Future Improvements

Possible future improvements include:

- 👤 Personalized recommendations based on user preferences
- ⭐ User ratings and watch history
- 🔥 Trending anime integration
- 🌐 Integration with live anime databases
- 🎬 Streaming-platform availability
- 💬 Conversational recommendation interface
- 📈 Improved recommendation evaluation
- ⚡ Faster inference and retrieval
- 📱 Improved mobile interface

---

## 👨‍💻 Author

**Soyam Bais**

IIT Patna

---

## 📜 License

This project is created for educational and portfolio purposes.