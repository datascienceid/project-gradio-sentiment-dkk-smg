import gradio as gr

import json
import string
import re
import functools
import operator

import pandas as pd

import emoji

from nltk import word_tokenize

from joblib import load


# CLEANSING
def cleansing(data):
    # lowercasing
    data = data.lower()
    # remove punctuation
    punct = string.punctuation
    translator = str.maketrans(punct, ' '*len(punct))
    data = data.translate(translator)
    # remove newline
    data = data.replace('\n', ' ')
    # remove digit
    pattern = r'[0-9]'
    data = re.sub(pattern, '', data)
    # remove extra space
    data = ' '.join(data.split())
    return data

# CONVERT EMOJIS
df_emoji = pd.read_csv('emoji_to_text.csv')
UNICODE_EMO = {row['emoji']:row['makna'] for idx,row in df_emoji.iterrows()}
def convert_emojis(text):
    # split emojis
    em_split_emoji = emoji.get_emoji_regexp().split(text)
    em_split_whitespace = [substr.split() for substr in em_split_emoji]
    em_split = functools.reduce(operator.concat, em_split_whitespace)
    text = ' '.join(em_split)
    # convert emojis
    for emot in UNICODE_EMO:
        text = re.sub(r'('+emot+')', "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()), text)
    return text.lower()

# NORMALIZE COLLOQUIAL/ALAY
kamus_alay = json.load(open('kamus_alay.json', 'r'))
def normalize_text(data):
    word_tokens = word_tokenize(data)
    result = [kamus_alay.get(w,w) for w in word_tokens]
    return ' '.join(result)

# REMOVE STOPWORDS
stop_words = [sw.strip() for sw in open('stop_words.txt', 'r').readlines()]
def remove_stopword(text, stop_words=stop_words):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

# PREPROCESS PIPELINE
def preprocess(text):
    text = cleansing(text)
    text = convert_emojis(text)
    text = normalize_text(text)
    text = remove_stopword(text)
    return text

# PREDICT SENTIMENT
def predict_sentiment(text):
    text = preprocess(text)

    # load tf-idf vectorizer model
    vectorizer = load('tfidf-vectorizer.joblib')
    feature = vectorizer.transform([text])

    # load model SVC
    svc = load("tfidf_svc_tuned.joblib")
    pred = svc.predict_proba(feature)[0]

    return {'Neutral': pred[0], 'Positive': pred[1], 'Negative': pred[2]}

# sample_text1 = "Ayooo.. Tetep ProKes ketat..!!! Janhan lengah..!!! Semangat...!!!"
gr.Interface(
    fn=predict_sentiment,
    title="Analisis Sentimen Komentar Instagram 🤗",
    description="Isikan kolom text dengan komentar, kemudian biarkan model machine learning memprediksikan hasil sentimen untukmu!",
    inputs=gr.inputs.Textbox(lines=7, label="Text"),
    outputs="label").launch()