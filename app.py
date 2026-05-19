from flask import Flask, render_template, request
import pandas as pd
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textblob import TextBlob
from fuzzywuzzy import process as fuzzy_process

from interview_data import categories


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)


data = pd.read_csv("faq.csv", on_bad_lines='skip')

questions = data['question'].tolist()
answers   = data['answer'].tolist()


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def correct_spelling(text):
    """Auto-correct spelling errors using TextBlob."""
    try:
        return str(TextBlob(text).correct())
    except Exception:
        return text

def preprocess(text):
    """Lowercase → spell-correct → tokenize → remove stopwords/punct → lemmatize."""
    text   = text.lower()
    text   = correct_spelling(text)             
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)            
        for word in tokens
        if word not in stop_words
        and word not in string.punctuation
        and word.isalpha()                      
    ]
    return " ".join(tokens)


processed_questions = [preprocess(q) for q in questions]

vectorizer       = TfidfVectorizer(ngram_range=(1, 2))   
question_vectors = vectorizer.fit_transform(processed_questions)


def get_response(user_input):
    """
    1. Preprocess & spell-correct the input.
    2. Try TF-IDF cosine similarity (primary).
    3. Fall back to fuzzy string matching (catches creative spelling).
    4. Return a helpful fallback message if neither is confident.
    """

    if not user_input or not user_input.strip():
        return "Please type a question so I can help you! "

    processed_input = preprocess(user_input)

    
    try:
        input_vector = vectorizer.transform([processed_input])
        similarity   = cosine_similarity(input_vector, question_vectors)
        best_idx     = similarity.argmax()
        score        = similarity[0][best_idx]
    except Exception:
        score = 0

    if score >= 0.2:                             
        return answers[best_idx]

   
    corrected_input = correct_spelling(user_input)
    fuzzy_result    = fuzzy_process.extractOne(
        corrected_input,
        questions,
        score_cutoff=55                        
    )

    if fuzzy_result:
        matched_question, fuzzy_score, matched_idx = fuzzy_result
        return answers[matched_idx]

    
    return (
        "I'm not sure I understood that. Could you rephrase your question? "
        "I can answer questions about Java, OOP, DBMS, DSA, Python, OS, "
        "Networking, and HR interview topics. "
    )


chat_history      = []
current_question  = 0
selected_category = "Java"



@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form.get("message", "").strip()
        if user_input:
            response = get_response(user_input)
            chat_history.append({"user": user_input, "bot": response})

    return render_template("index.html", chats=chat_history)


@app.route("/clear")
def clear_chat():
    global chat_history
    chat_history = []
    return render_template("index.html", chats=chat_history)


@app.route("/interview", methods=["GET", "POST"])
def interview():
    global current_question, selected_category

    feedback = ""

    if request.method == "POST":

        if "category" in request.form:
            selected_category = request.form["category"]
            current_question  = 0

        else:
            user_answer  = request.form.get("answer", "").lower()
            current_data = categories[selected_category]
            keywords     = current_data[current_question]["keywords"]

            # Count keyword matches (whole-word, not substring)
            answer_words = set(user_answer.split())
            score = sum(1 for kw in keywords if kw in answer_words)

            total = len(keywords)
            if score >= max(2, total // 2):
                feedback = " Great answer! You covered the important concepts well."
            elif score == 1:
                feedback = " Good attempt! Try adding more technical depth to your answer."
            else:
                feedback = " Keep practising! Review the core concepts and try again."

            current_question = (current_question + 1) % len(current_data)

    current_data = categories[selected_category]
    question     = current_data[current_question]["question"]

    return render_template(
        "interview.html",
        question=question,
        feedback=feedback,
        categories=categories.keys(),
        selected_category=selected_category,
    )


if __name__ == "__main__":
    app.run(debug=True)