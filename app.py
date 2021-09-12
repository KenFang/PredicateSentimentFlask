import os
import pickle
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import everygrams

import nltk

from flask import Flask, request

app = Flask(__name__)


def extract_feature(document):
    nltk.data.path = ['/app/nltk_data/']

    stopwords_eng = stopwords.words('english')

    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(document)
    lemmas = [str(lemmatizer.lemmatize(w)) for w in words if w not in stopwords_eng and w not in punctuation]
    document = " ".join(lemmas)
    document = document.lower()
    document = re.sub(r'[^a-zA-Z0-9\s]', ' ', document)
    words = [w for w in document.split(" ") if w != "" and w not in stopwords_eng and w not in punctuation]
    return [str(" ".join(ngram)) for ngram in list(everygrams(words, max_len=3))]


def bag_of_words(document):
    bag = {}
    for w in document:
        bag[w] = bag.get(w, 0) + 1
    return bag


def get_predicate_model():
    model_file = open('/app/sa_classifier.pickle', 'rb')
    model = pickle.load(model_file)
    model_file.close()
    return model


def get_sentiment(review):
    words = extract_feature(review)
    words = bag_of_words(words)
    model = get_predicate_model()
    return model.classify(words)


@app.route('/predict', methods=['GET', 'POST'])
def predict():  # put application's code here
    if request.method == 'GET':
        words = request.args.get('review')
    else:
        words = request.get_json(force=True)['review']

    if not words:
        return 'No input provided'

    return get_sentiment(words)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
