# 🎬 Movie Genre Prediction 

## 📌 Project Overview
This project is a Machine Learning based web application that predicts the genre of a movie
based on its plot summary. It uses Natural Language Processing (NLP) techniques to analyze
textual data and classify it into appropriate movie genres.

This project was developed as part of the CodSoft Machine Learning Internship.

---

## 🎯 Objective
- Predict the genre of a movie using its plot description
- Apply TF-IDF vectorization for text feature extraction
- Build an interactive web application using Streamlit
- Demonstrate an end-to-end ML workflow

---

## 📂 Dataset
- Source: Kaggle
- Dataset Name: IMDb Genre Classification Dataset
- Description:
  The dataset contains movie plot summaries along with their corresponding genres.

### Dataset Format
Each record follows this structure:

ID ::: GENRE ::: MOVIE_PLOT

Example:
1 ::: drama ::: A young man struggles to survive in society...

---

## 🛠️ Technologies Used
- Programming Language: Python
- Libraries & Tools:
  - Streamlit
  - Scikit-learn
  - Pandas
  - NumPy
  - Joblib
- Machine Learning Techniques:
  - TF-IDF Vectorization
  - Text Classification (Naive Bayes / Logistic Regression)

---

## ⚙️ Project Workflow
1. Loaded and preprocessed IMDb movie plot data
2. Converted text data into numerical features using TF-IDF
3. Trained a machine learning classifier
4. Saved the trained model and vectorizer
5. Built a Streamlit-based web interface
6. Predicted movie genres in real time

---

## 🌐 Web Application Features
- Clean and modern UI
- Dark theme interface
- Input validation (minimum 5 words)
- Example plot provided
- Real-time genre prediction

---

## 🧪 Example

Input:
A detective investigates a series of mysterious murders in the city.

Output:
Predicted Genre: Thriller

---

## 📁 Project Structure

movie_genre_prediction/
│
├── streamlit_app.py
├── model/
│   ├── genre_model.pkl
│   └── tfidf.pkl
├── train_data.txt
└── README.md

---
## 📊 Model Evaluation

The model was evaluated using standard classification metrics.

- Accuracy: ~85% (may vary based on training data)
- Precision, Recall, and F1-score were calculated using a classification report
- The model performs well on dominant genres such as Drama, Thriller, and Horror

Evaluation Metrics Used:
- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

## ▶️ How to Run the Project

### Step 1: Install Dependencies
pip install streamlit scikit-learn pandas numpy joblib

### Step 2: Run the Application
streamlit run streamlit_app.py

### Step 3: Open in Browser
http://localhost:8501

---

## 🚀 Future Enhancements
- Add prediction confidence scores
- Support multi-genre classification
- Improve accuracy using advanced NLP models
- Deploy the application online
- Enhance UI with additional visual elements

---

## ✅ Conclusion
This project demonstrates how machine learning and NLP techniques can be combined with a
web-based interface to build a practical movie genre prediction system. It highlights
skills in data preprocessing, feature extraction, model training, and deployment.

---

