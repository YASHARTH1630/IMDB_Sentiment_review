import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load model
model = load_model('simple_rnn_imdb.h5')

# Load IMDB word index
word_index = imdb.get_word_index()

# Reverse word index
reverse_word_index = {value: key for key, value in word_index.items()}

VOCAB_SIZE = 10000
maxlen = 500

# Preprocess function
def preprocess_text(text):

    words = text.lower().split()

    sequence = []

    for word in words:

        index = word_index.get(word)

        if index is not None:

            index = index + 3

            if index < VOCAB_SIZE:
                sequence.append(index)
            else:
                sequence.append(2)

        else:
            sequence.append(2)

    padded = pad_sequences([sequence], maxlen=maxlen)

    return padded


# Prediction function
def predict_sentiment(review):

    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return sentiment, prediction[0][0]


# Streamlit UI
st.title("IMDB Movie Review Sentiment Analysis")

st.write("Enter a movie review below:")

user_input = st.text_area("Movie Review")


if st.button("Classify"):

    sentiment, score = predict_sentiment(user_input)

    st.write(f"Sentiment: {sentiment}")

    st.write(f"Prediction Score: {score}")