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