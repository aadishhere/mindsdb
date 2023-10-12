import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from keras.losses import SparseCategoricalCrossentropy


class AyurvedicChatbotRNN:
    def __init__(self, symptoms, remedies):
        self.symptoms = symptoms
        self.remedies = remedies
        self.tokenizer = Tokenizer()
        self.total_words = None
        self.max_sequence_length = None
        self.model = None

    def preprocess_data(self):
        # Tokenize input symptoms
        self.tokenizer.fit_on_texts(self.symptoms)
        self.total_words = len(self.tokenizer.word_index) + 1
        input_sequences = self.tokenizer.texts_to_sequences(self.symptoms)
        self.max_sequence_length = max([len(x) for x in input_sequences])
        input_sequences = pad_sequences(input_sequences, maxlen=self.max_sequence_length, padding='pre')

        # Tokenize output remedies
        output_sequences = self.tokenizer.texts_to_sequences(self.remedies)
        output_sequences = pad_sequences(output_sequences, maxlen=self.max_sequence_length, padding='pre')

        return input_sequences, output_sequences

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.total_words, 100, input_length=self.max_sequence_length))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100))
        self.model.add(Dense(self.total_words, activation='softmax'))

        self.model.compile(loss=SparseCategoricalCrossentropy, optimizer='adam')

    def train_model(self, input_sequences, output_sequences, epochs=50):
        self.model.fit(input_sequences, np.array(output_sequences), epochs=epochs, verbose=1)

    def save_model(self, model_path='rnn_model.h5', tokenizer_path='tokenizer.json'):
        self.model.save(model_path)
        with open(tokenizer_path, 'w') as tokenizer_file:
            json.dump(self.tokenizer.to_json(), tokenizer_file)

    def load_model(self, model_path='rnn_model.h5', tokenizer_path='tokenizer.json'):
        with open(tokenizer_path, 'r') as tokenizer_file:
            tokenizer_json = json.load(tokenizer_file)
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
        self.model = tf.keras.models.load_model(model_path)

