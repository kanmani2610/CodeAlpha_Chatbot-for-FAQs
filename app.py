from flask import Flask, render_template, request
import pandas as pd
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from interview_data import categories

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

# Load FAQ data
data = pd.read_csv("faq.csv")

questions = data['question'].tolist()
answers = data['answer'].tolist()

# Preprocess text
def preprocess(text):

    text = text.lower()

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stopwords.words('english')
        and word not in string.punctuation
    ]

    return " ".join(tokens)

# Process FAQ questions
processed_questions = [preprocess(q) for q in questions]

# Convert text into vectors
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(processed_questions)

# Chatbot response function
def get_response(user_input):

    processed_input = preprocess(user_input)

    input_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(input_vector, question_vectors)

    best_match_index = similarity.argmax()

    score = similarity[0][best_match_index]

    if score < 0.3:
        return "Sorry, I could not understand your question properly."

    return answers[best_match_index]

# Chat history
chat_history = []

# Interview question index
current_question = 0
selected_category = "Java"

# Home Route
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        user_input = request.form["message"]

        response = get_response(user_input)

        chat_history.append({
            "user": user_input,
            "bot": response
        })

    return render_template(
        "index.html",
        chats=chat_history
    )

# Clear Chat Route
@app.route("/clear")
def clear_chat():

    global chat_history

    chat_history = []

    return render_template(
        "index.html",
        chats=chat_history
    )

# Interview Mode
@app.route("/interview", methods=["GET", "POST"])
def interview():

    global current_question
    global selected_category

    feedback = ""

    if request.method == "POST":

        if "category" in request.form:

            selected_category = request.form["category"]

            current_question = 0

        else:

            user_answer = request.form["answer"].lower()

            current_data = categories[selected_category]

            keywords = current_data[current_question]["keywords"]

            score = 0

            for word in keywords:

                if word in user_answer:
                    score += 1

            if score >= 2:

                feedback = "✅ Great answer! You covered important concepts."

            elif score == 1:

                feedback = "⚠️ Good attempt, but add more technical depth."

            else:

                feedback = "❌ Try improving your answer with better concepts."

            current_question = (
                current_question + 1
            ) % len(current_data)

    current_data = categories[selected_category]

    question = current_data[current_question]["question"]

    return render_template(
        "interview.html",
        question=question,
        feedback=feedback,
        categories=categories.keys(),
        selected_category=selected_category
    )

if __name__ == "__main__":
    app.run(debug=True)