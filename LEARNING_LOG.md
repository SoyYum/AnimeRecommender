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