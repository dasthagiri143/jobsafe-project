# JobSafe — Fake Job Posting Detector

JobSafe is a mini-project that detects fake job postings using machine learning (Logistic Regression + TF-IDF).

---

## 📂 Project Structure

JobSafe_Project/
├── data/ # Input dataset
├── jobsafe/ # ML code + saved model
├── app.py # Flask API
├── requirements.txt # Dependencies
├── test_predict.sh # Quick curl test script
├── index.html + script.js # Simple frontend
├── README.md
├── DEPLOY.md
└── venv/ # Virtual environment (optional)

---

## ✅ How to Run Locally

1️⃣ **Create & activate virtual env**
```bash
python3 -m venv venv
source venv/bin/activate