from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "..", "model", "jobsafe_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "..", "model", "vectorizer.pkl"), "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = " ".join([
        request.form["title"],
        request.form["location"],
        request.form["company_profile"],
        request.form["description"],
        request.form["requirements"]
    ])
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    result = "Fake Job Ad ❌" if prediction == 1 else "Real Job ✅"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
