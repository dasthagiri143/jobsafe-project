import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("data/fake_job_postings.csv")

# Drop missing values and keep only required columns
df = df[["title", "description", "fraudulent"]].dropna()

# Combine title and description
df["text"] = df["title"] + " " + df["description"]

# Balance the dataset
real = df[df["fraudulent"] == 0]
fake = df[df["fraudulent"] == 1]
fake_upsampled = resample(fake, replace=True, n_samples=len(real), random_state=42)
balanced_df = pd.concat([real, fake_upsampled])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    balanced_df["text"], balanced_df["fraudulent"], test_size=0.2, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/jobsafe_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully.")
