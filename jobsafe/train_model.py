# train_model.py
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
df = pd.read_csv(os.path.join(BASE_DIR, 'data/fake_job_postings.csv'))

# Drop NaNs and duplicates
df = df.dropna(subset=['description', 'fraudulent']).drop_duplicates()

# Balance the dataset (optional: downsample real jobs)
fake = df[df['fraudulent'] == 1]
real = df[df['fraudulent'] == 0].sample(n=len(fake), random_state=42)
df_balanced = pd.concat([fake, real])

# Features and labels
X = df_balanced['description']
y = df_balanced['fraudulent']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Model training
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
with open(os.path.join(BASE_DIR, 'model/jobsafe_model.pkl'), 'wb') as f:
    pickle.dump(model, f)
with open(os.path.join(BASE_DIR, 'model/vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully.")
