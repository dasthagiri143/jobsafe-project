# JobSafe – A Machine Learning Approach to Identify Fake Job Ads

JobSafe is a machine learning-based web application built with Flask that detects fake job advertisements using NLP techniques and classification models.

## 🚀 Features

- 🔍 Predicts whether a job ad is real or fake
- 🧠 Trained using logistic regression on cleaned and preprocessed job ad data
- 💻 Simple Flask-based web UI for predictions
- 📊 Includes balanced dataset handling and model persistence

---

## 📁 Project Structure

jobsafe_project/
│
├── app/
│ ├── main.py # Flask app
│ ├── templates/
│ │ └── index.html # Frontend HTML form
│ └── static/
│ └── style.css # Basic styling
│
├── data/
│ └── fake_job_postings.csv # Dataset used for training
│
├── model/
│ ├── jobsafe_model.pkl # Trained ML model
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer
│
├── train_model.py # Model training script
├── requirements.txt # Python dependencies
└── README.md # Project overview

yaml
Copy code

---

## ⚙️ Installation & Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/dasthagiri143/jobsafe-project.git
cd jobsafe-project
Step 2: Create and Activate a Virtual Environment
bash
Copy code
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
bash
Copy code
pip install -r requirements.txt
Step 4: Train the Model
bash
Copy code
python train_model.py
This saves the model and vectorizer to the model/ directory.

Step 5: Run the App
bash
Copy code
python app/main.py
Then open your browser and visit:
http://127.0.0.1:5000

💡 Example Job Ads
✅ Real Job Example:
css
Copy code
We are hiring a Python Developer to work with our engineering team to develop and maintain high-quality software. Must have experience in Django and REST APIs.
❌ Fake Job Example:
pgsql
Copy code
Work from home! Earn ₹50,000 weekly by typing simple words. No skills needed. Click here to register: http://scamjob.link
📊 Model Used
TF-IDF Vectorizer

Logistic Regression Classifier

Balanced dataset (resampled)

Accuracy: ~90% on test set

🧠 Future Scope
Integration with real job portals (e.g., LinkedIn, Naukri)

Deep learning models like LSTM or BERT

Browser extension for live job detection

👨‍💻 Author
Dasthagiri Gagguturu

GitHub Profile
