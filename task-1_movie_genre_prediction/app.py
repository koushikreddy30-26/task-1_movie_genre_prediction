import streamlit as st
import joblib
st.set_page_config(
    page_title="Movie Genre Prediction",
    page_icon="🎬",
    layout="centered"
)
st.markdown("""
<style>
.block-container {
    max-width: 750px;
    padding-top: 2rem;
}
h1 {
    text-align: center;
}
.card {
    background-color: #1f2933;
    padding: 1.2rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.result {
    background-color: #14532d;
    padding: 1rem;
    border-radius: 8px;
    font-size: 18px;
    margin-top: 1rem;
}
textarea {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)
model = joblib.load("task-1_movie_genre_prediction/model/genre_model.pkl")
tfidf = joblib.load("task-1_movie_genre_prediction/model/tfidf.pkl")
st.title("🎬 Movie Genre Prediction")
st.markdown("""
<div class="card">
<h3>📝 About the Model</h3>
<p>
This application uses a <b>machine learning model trained on IMDb movie plot summaries</b>
to predict the <b>genre of a movie</b>.
It applies <b>Natural Language Processing (NLP)</b> techniques such as
<b>TF-IDF vectorization</b> and a classification algorithm.
</p>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class="card">
<h3>📌 How to Use</h3>
<ol>
<li>Enter a short movie plot (minimum 5 words)</li>
<li>Click <b>Predict Genre</b></li>
<li>View the predicted genre</li>
</ol>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class="card">
<h3>🎥 Example</h3>
<i>A detective investigates a series of mysterious murders in the city.</i>
</div>
""", unsafe_allow_html=True)
plot = st.text_area(
    "Enter movie plot",
    height=120,
    placeholder="Type the movie plot here..."
)
if st.button("Predict Genre", type="primary"):
    if len(plot.strip().split()) < 5:
        st.warning("⚠️ Please enter at least 5 meaningful words.")
    else:
        vector = tfidf.transform([plot])
        prediction = model.predict(vector)[0]

        st.markdown(
            f"<div class='result'>🎯 Predicted Genre: <b>{prediction.title()}</b></div>",
            unsafe_allow_html=True
        )

# =========================
# Footer
# =========================
st.markdown("""
<hr>
<p style="text-align:center; font-size:14px;">
Built by <b>Jayanth Reddy</b> | CodSoft Machine Learning Internship
</p>
""", unsafe_allow_html=True)
