import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

VOCAB_SIZE = 10000
MAX_LEN = 250
MODEL_PATH = 'sentiment_analysis_model.h5'

# Load the saved model
model = load_model(MODEL_PATH)

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def encode_texts(text_list):
    sequences = tokenizer.texts_to_sequences(text_list)
    return pad_sequences(sequences, maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)

def predict_sentiments(text_list):
    encoded_inputs = encode_texts(text_list)
    predictions = np.argmax(model.predict(encoded_inputs), axis=-1)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return [label_map[pred] for pred in predictions]
