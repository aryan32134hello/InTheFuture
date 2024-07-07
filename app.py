from flask import Flask, jsonify, render_template, request
from rag_func import main
from rag_func_for_pdf import main_2
import keras
import numpy as np


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/thinkbot")
def thinkbot():
    return render_template('rag_chain.html')


@app.route("/thinkbotpdf")
def thinkbotpdf():
    return render_template('rag_pdf.html')


@app.route("/translatebot")
def lang_trans():
    return render_template('lang_trans.html')


@app.route("/submit", methods=['POST'])
def submit():
    model_id = "google/gemma-2b-it"
    url = request.form["website"]
    question = request.form["question"]
    print(f"url is {url} and question is {question}")
    ans = my_main(url, question, model_id)
    print(f"The answer is {ans}")
    return jsonify({'ans': ans})


@app.route("/submit_pdf", methods=['POST'])
def submit_pdf():
    model_id = "google/gemma-2b-it"
    pdf_file = request.files['choose']
    question = request.form["question"]
    print(f"url is {pdf_file.name} and question is {question}")
    ans = my_main_2(pdf_file, question, model_id)
    print(f"The answer is {ans}")
    return jsonify({'ans': ans})


@app.route("/translate", methods=['POST'])
def translate():
    text = request.form["text"]
    language = request.form["language"]
    ans = ""
    print("getting")
    if language == "French":
        ans = getFrenchText(text)
    elif language == "Hindi":
        ans = getHindiText(text)
    # else :
    #     ans = getSpanishText(text)
    print(ans)
    return jsonify({'ans': ans})


def getFrenchText(text):
    answer = ""
    model = keras.models.load_model("./tf_model_french")
    for i in range(15):
        X_encoded = np.array([text])
        X_decoded = np.array(["sos " + answer])
        y_prob = model.predict((X_encoded, X_decoded), verbose=0)[0, i]
        y_prob_id = np.argmax(y_prob)
        predicted_word = model.tokenizer_fr.get_vocabulary()[y_prob_id]
        if predicted_word == "eos":
            break
        answer = answer + " " + predicted_word
    return answer.strip()


def getHindiText(text):
    print("getting model")
    answer = ""
    model = keras.models.load_model("./tf_model_hindi")
    for i in range(20):
        X_encoded = np.array([text])
        X_decoded = np.array(["sos " + answer])
        y_prob = model.predict((X_encoded, X_decoded), verbose=0)[0, i]
        y_prob_id = np.argmax(y_prob)
        predicted_word = model.tokenizer_hn.get_vocabulary()[y_prob_id]
        if predicted_word == "eos":
            break
        answer = answer + " " + predicted_word
    return answer.strip()


def my_main(url, question, model_id):
    ans = main(url, question, model_id)
    return ans


def my_main_2(pdf_file, question, model_id):
    ans = main_2(pdf_file, question, model_id)
    return ans

if __name__ == "__main__":
    app.run(debug=True)
