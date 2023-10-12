import os
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class AyurvedicChatbotRNN:
    def __init__(self, data_dir, preprocessed_dir):
        self.data_dir = data_dir
        self.preprocessed_dir = preprocessed_dir
        self.tokenizer = Tokenizer()
        self.total_words = None
        self.max_sequence_length = None

    def clean_and_tokenize(self, text):
        cleaned_text = re.sub(r'[^\w\s]', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def preprocess_data(self):
        texts = []
        for filename in os.listdir(self.data_dir):
            input_path = os.path.join(self.data_dir, filename)
            with open(input_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
                cleaned_text = self.clean_and_tokenize(raw_text)
                texts.append(cleaned_text)
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        self.total_words = len(self.tokenizer.word_index) + 1
        self.max_sequence_length = max([len(x) for x in sequences])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='pre')
        return padded_sequences

    def save_tokenizer(self, tokenizer_path='tokenizer.json'):
        tokenizer_json = self.tokenizer.to_json()
        with open(tokenizer_path, 'w') as tokenizer_file:
            tokenizer_file.write(tokenizer_json)


if __name__ == "__main__":
    DATA_DIR = 'data/ayurvedic_books/'
    PREPROCESSED_DIR = 'data/preprocessed_data/'

    chatbot = AyurvedicChatbotRNN(DATA_DIR, PREPROCESSED_DIR)
    input_sequences = chatbot.preprocess_data()
    chatbot.save_tokenizer()
    print('Data preprocessed and tokenizer saved.')
