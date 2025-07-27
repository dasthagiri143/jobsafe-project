import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/fake_job_postings.csv")

# Drop nulls and balance the classes
df = df[['title', 'location', 'company_profile', 'description', 'requirements', 'fraudulent']].dropna()
df = df[df['fraudulent'].isin([0, 1])]

# Combine text fields
df["text"] = df['title'] + " " + df['location'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements']

# Features and labels
X = df["text"]
y = df["fraudulent"]

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/jobsafe_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
