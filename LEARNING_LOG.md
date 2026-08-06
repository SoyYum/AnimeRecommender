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