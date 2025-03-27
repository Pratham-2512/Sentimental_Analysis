import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import pandas as pd
from fastapi import FastAPI
import uvicorn
import pickle

# Load dataset (Using IMDB dataset from TensorFlow Datasets)
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Pad sequences
train_data = pad_sequences(train_data, maxlen=250)
test_data = pad_sequences(test_data, maxlen=250)

# Build the Model
model = Sequential([
    Embedding(10000, 16),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
model.fit(train_data, train_labels, epochs=3, validation_data=(test_data, test_labels), batch_size=64)

# Save Model
model.save('sentiment_model.keras')  # Instead of 'my_model.h5'

# Create FastAPI App
app = FastAPI()

# Load Tokenizer (to preprocess new inputs)
tokenizer = Tokenizer(num_words=10000)

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=250)
    return padded

# Load Model
model = keras.models.load_model("sentiment_model.h5")

@app.post("/predict")
def predict_sentiment(text: str):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return {"sentiment": sentiment, "confidence": float(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

import requests

response = requests.get(uvicorn.run(app, host="127.0.0.1", port=8000))
print(response.json())
