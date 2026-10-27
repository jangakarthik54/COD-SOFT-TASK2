"""
Movie Rating Prediction using Python
Dataset: IMDb Movies India.csv inside archive (1).zip

Install:
    python -m pip install pandas scikit-learn joblib

Run:
    python movie_rating_prediction.py

The model predicts IMDb-style movie ratings using genre, director, actors,
year, duration, and vote count. Rows without a known rating are excluded.
"""
from pathlib import Path
import zipfile
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = BASE_DIR / "archive (1).zip"
CSV_PATH = BASE_DIR / "IMDb Movies India.csv"

# Load the dataset either from the ZIP file or from an extracted CSV.
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH, encoding="latin1")
elif ZIP_PATH.exists():
    with zipfile.ZipFile(ZIP_PATH) as archive:
        csv_files = [name for name in archive.namelist()
                     if name.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV file found inside the ZIP archive.")
        with archive.open(csv_files[0]) as file:
            df = pd.read_csv(file, encoding="latin1")
else:
    raise FileNotFoundError(
        "Put 'archive (1).zip' or 'IMDb Movies India.csv' "
        "in the same folder as this script."
    )

# Clean column names and remove rows without a known target rating.
df.columns = df.columns.str.strip()
if "Rating" not in df.columns:
    raise ValueError("The dataset must contain a 'Rating' column.")

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df = df.dropna(subset=["Rating"]).copy()

# Convert numeric-looking columns to numbers.
numeric_features = ["Year", "Duration", "Votes"]
categorical_features = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]

for col in numeric_features:
    if col in df.columns:
        # Handles values such as '120 min' and vote counts containing commas.
        df[col] = (
            df[col].astype("string")
            .str.replace(",", "", regex=False)
            .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = pd.NA

for col in categorical_features:
    if col not in df.columns:
        df[col] = "Unknown"
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    df.loc[df[col].isin(["", "nan", "None", "<NA>"]), col] = "Unknown"

feature_columns = numeric_features + categorical_features
X = df[feature_columns]
y = df["Rating"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Total movies with ratings: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print("\nTraining model...")

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.3f} rating points")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R²): {r2:.3f}")

model_path = BASE_DIR / "movie_rating_model.joblib"
joblib.dump(model, model_path)
print(f"\nSaved model to: {model_path}")

# Example prediction for a hypothetical movie.
new_movie = pd.DataFrame([{
    "Year": 2024,
    "Duration": 120,
    "Votes": 1000,
    "Genre": "Drama, Thriller",
    "Director": "Unknown Director",
    "Actor 1": "Unknown Actor",
    "Actor 2": "Unknown Actor",
    "Actor 3": "Unknown Actor",
}])

predicted_rating = model.predict(new_movie[feature_columns])[0]
predicted_rating = max(0, min(10, predicted_rating))
print("\n--- Example Prediction ---")
print(f"Estimated movie rating: {predicted_rating:.2f}/10")
print("(This is a sample prediction, not an actual rating.)")
