# Day 1

- Learned how Git tracks project history using commits.
- Understood the purpose of virtual environments.
- Learned why `.gitignore` and `requirements.txt` are essential.
- Understood the difference between content-based and collaborative recommendation systems.
- Decided to build a semantic anime recommendation engine using contextual embeddings.

# Day 2

- Learned why NumPy is the foundation of ML.
- Understood the difference between Python lists and NumPy arrays.
- Learned vectorized operations.
- Learned array properties: shape, size, ndim and dtype.
- Learned 2D arrays and indexing.

# Day 3 - Pandas & Dataset Exploration

## What I learned
- Loaded a real-world anime dataset using Pandas.
- Learned to inspect datasets using:
  - df.head()
  - df.shape
  - df.columns
  - df.info()
  - df.describe()
- Understood the difference between numerical and categorical columns.
- Learned why missing values are important before training ML models.
- Identified the columns needed for the first version of the recommender.

## Dataset Summary
- Rows: 30,075
- Columns: 29

## Key Columns for Version 1
- title
- synopsis
- genres
- score
- episodes
- type
- image_url

## Challenges
- Initially ran the script using VS Code's Code Runner, which used the wrong Python interpreter.
- Fixed it by running the script from the terminal using the virtual environment.

## Next Steps
- Clean the dataset.
- Handle missing values.
- Remove unnecessary columns.

# Day 4 - Data Cleaning

## What I learned
- Selected only the columns needed for the recommender.
- Checked missing values using `df.isnull().sum()`.
- Removed anime with missing synopses since the recommender is plot-based.
- Learned the difference between duplicate titles and duplicate rows.
- Removed only exact duplicate rows using `drop_duplicates()`.
- Saved the cleaned dataset as `cleaned_anime.csv`.

## Dataset Summary
- Original dataset: 30,075 rows
- After removing missing synopses: 25,037 rows
- After removing exact duplicates: 24,905 rows

## Key Takeaways
- Don't remove data without understanding why.
- Missing values should only be handled if they affect the model.
- Duplicate titles are not always duplicate anime.

## Next Steps
- Preprocess the synopsis text.
- Prepare the text for TF-IDF vectorization.

# Day 5 – NLP Preprocessing

## Learned
- Understood why NLP preprocessing is important for recommendation systems.
- Learned tokenization using split().
- Learned what stop words are and why they are removed.
- Used NLTK's English stop word corpus.
- Learned the basics of lemmatization using WordNetLemmatizer.
- Cleaned an anime synopsis by converting to lowercase, removing stop words, and lemmatizing words.

## Key Takeaways
- Raw text cannot be directly used by ML models.
- Preprocessing removes noise and preserves meaningful information.
- Simple split() leaves punctuation attached, so better tokenization is needed.
- Default lemmatization assumes nouns, so verb forms require POS information.

## Day 6 - TF-IDF Recommendation Engine

### What I Learned
- Learned how TF-IDF converts text into numerical vectors.
- Understood the difference between TF and IDF and why common words receive lower importance.
- Learned how `TfidfVectorizer` builds a vocabulary and transforms every synopsis into a sparse TF-IDF vector.
- Understood why sparse matrices are memory efficient for NLP tasks.
- Learned how cosine similarity measures the similarity between two vectors using the angle between them.
- Learned how `enumerate()`, `sorted()`, and `lambda` can be used to rank recommendations.

### What I Built
- Generated a TF-IDF matrix from all cleaned anime synopses.
- Computed pairwise cosine similarity between every anime.
- Built the first version of a content-based anime recommendation engine.
- Given an anime title, the program finds and displays the top 10 most similar anime.

### Challenges Faced
- Understanding how TF-IDF vectors are constructed from the entire vocabulary instead of individual synopsis length.
- Understanding how cosine similarity compares vectors.
- Interpreting the similarity matrix and using `enumerate()` to preserve indices while sorting.

### Next Steps
- Convert the recommendation logic into reusable functions.
- Accept user input instead of hardcoded titles.
- Improve title searching and handle invalid inputs.

## Day 7 - Improving the Recommendation Engine

### What I Learned
- Refactored the recommendation logic into a reusable function.
- Learned the importance of separating recommendation logic from user interaction.
- Implemented case-insensitive title searching using Pandas string methods.
- Added partial title matching using `str.contains()`.
- Learned how to handle multiple matching results by allowing the user to choose the intended anime.
- Practiced returning multiple values from a function using tuple unpacking.

### What I Built
- Converted the recommender into a reusable function.
- Added support for user input instead of hardcoded anime names.
- Implemented case-insensitive and partial title search.
- Added interactive selection when multiple anime titles match the search.
- Improved the overall usability of the recommendation engine.

### Challenges Faced
- Understanding how to separate recommendation logic from the user interface.
- Handling multiple matching titles without making incorrect assumptions.
- Designing function return values to communicate different outcomes cleanly.

### Next Steps
- Build a web interface for the recommender.
- Display recommendations in a browser instead of the terminal.
- Prepare the recommendation engine for integration into a web application.

## Day 8 – Improving Search Experience

### What I Learned
- Learned the concept of fuzzy searching and why it is useful when users make spelling mistakes.
- Explored Python's `difflib` module and understood its limitations on large datasets.
- Learned to use the `RapidFuzz` library for approximate string matching.
- Understood similarity scores and different scorers such as `QRatio` and `WRatio`.
- Realized that fuzzy search is best used as a fallback rather than the primary search method.
- Refactored the recommender by separating the search logic into a dedicated `search_anime()` function.

### Improvements Made
- Added partial title search.
- Added fuzzy search suggestions when no exact or partial match exists.
- Filtered low-confidence suggestions.
- Improved the overall search pipeline and made the code cleaner and more modular.

### Challenges Faced
- Both `difflib` and `RapidFuzz` produced unexpected suggestions for some typos on a dataset containing over 25,000 anime titles.
- Learned the importance of evaluating libraries on real datasets instead of assuming they always produce ideal results.

## Day 9 - Feature Engineering

### What I learned
- Feature engineering improves the information given to the model.
- Combined multiple metadata fields into one text feature.
- TF-IDF can vectorize any text, not just the synopsis.
- Refactored the recommender into smaller functions.
- Learned why separating search and recommendation logic leads to cleaner code.

### Improvements
- Updated cleaned_anime.csv to include:
  - genres
  - themes
  - demographics
  - source
  - studios
- Added combined_features.
- Created recommend_from_index().
- Fixed the multiple match bug.

## Day 10 - Feature Engineering & Recommender Finalization

### What I did:
- Built combined feature column using:
  - genres (weighted x3)
  - themes (x1)
  - demographics, type, source, studios
  - synopsis
- Tuned TF-IDF vectorizer:
  - ngram_range = (1, 2)
  - stop_words = "english"
  - max_df = 0.8
  - min_df = 2
  - max_features = 5000
- Generated similarity matrix using cosine similarity
- Implemented recommendation system:
  - top 10 similar anime
- Built search system:
  - exact match using substring
  - fuzzy match using RapidFuzz (QRatio)
- Handled edge cases:
  - multiple matches → user selection
  - fuzzy suggestions
  - not found case
- Improved recommendation quality by tuning feature weights
- Refactored code:
  - separated logic into functions
  - created cleaner pipeline

### Key Learnings:
- Feature weighting has huge impact on recommendation quality
- Too much synopsis makes results noisy
- Genres provide strong signal → need higher weight
- TF-IDF tuning (ngrams, filtering) improves similarity quality
- Fuzzy matching needs to be controlled (WRatio was too aggressive)
- Clean structure makes debugging much easier

### Results:
- Recommendations feel accurate and meaningful
- System works for real-world anime queries
- Stable CLI-based recommender completed

# Day 12 – CLI Interface

## Learned
- Used sys.argv to take input from terminal
- Difference between CLI input and input()
- Built dual input system (CLI + interactive)
- Separated logic (recommender.py) from execution (main.py)

## Built
- CLI tool:
  python main.py Naruto
- Fallback:
  python main.py → asks input
- Handles all cases (fuzzy, multiple, success, not found)

## Takeaways
- CLI > input() for real applications
- Supporting both is best
- Usability matters, not just model

## Result
- Recommender is now a usable tool from terminal

# Day 13 – Feature Engineering + Model Improvement

## Learned
- Importance of feature weighting in content-based systems
- How TF-IDF reacts to noisy/common words
- Why plot/synopsis is useful but can introduce noise
- Using n-grams (1,2) to capture better context
- How stopwords affect recommendation quality

## Built
- Improved combined_features column:
  - Repeated genres to increase importance
  - Added themes, demographics, type, source, studios, synopsis
- Tuned TF-IDF:
  - ngram_range = (1,2)
  - max_df = 0.8
  - min_df = 2
  - max_features = 5000
- Fixed stop_words error (used list instead of invalid type)
- Rebuilt similarity model successfully

## Takeaways
- Feature engineering > model complexity
- Weighting important fields (like genres) improves relevance
- Too much raw text (synopsis) can hurt results if not cleaned
- Debugging sklearn errors is mostly about parameter types

# Day 14 - Semantic Embeddings + Hybrid Recommendation

## Learned
- Understood sentence embeddings using SBERT (Sentence-BERT)
- Learned the difference between TF-IDF and semantic embeddings
- Learned how cosine similarity works with embedding vectors
- Understood how batch_size helps process large datasets efficiently

## Built
- Installed and used sentence-transformers library
- Generated SBERT embeddings using all-MiniLM-L6-v2
- Created embeddings for all anime plots/features
- Saved anime embeddings for future recommendations
- Combined TF-IDF similarity and SBERT similarity using weighted scoring

Hybrid scoring:
- 40% TF-IDF similarity
- 60% SBERT similarity

- Updated recommender to rank using combined similarity scores
- Added English title display support
- Improved same-series filtering to avoid recommending sequels/movies of the same anime

## Takeaways
- TF-IDF understands keyword importance
- SBERT understands semantic meaning
- Combining both gives better recommendations than using either alone
- Embeddings allow machines to compare meaning instead of only matching words

## Result
- Recommender upgraded from keyword-based similarity to a hybrid semantic recommendation system
- Anime recommendations are now based on both plot meaning and important keywords

# Day 15

## Completed
- Improved recommendation filtering to reduce same-series recommendations.
- Added title normalization and series detection logic.
- Fixed recommendation ranking pipeline using combined TF-IDF + SBERT scores.
- Tested recommender on multiple anime inputs.

## Current Status
- Anime recommender pipeline is working.
- Search, embedding similarity, ranking, and filtering are functional.

## Issues / Next Steps
- Improve recommendation diversity.
- Explore better reranking methods like MMR.
- Improve franchise detection.