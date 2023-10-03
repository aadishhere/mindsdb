import os
import re
import pandas as pd
from nltk.tokenize import word_tokenize

# Constants
DATA_DIR = 'data/ayurvedic_books/'
PREPROCESSED_DIR = 'data/preprocessed_data/'
MINDSDB_MODEL_DIR = 'models/mindsdb_model/'

# Function to clean and tokenize the text data
def clean_and_tokenize(text):
    # Remove special characters and multiple spaces
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Tokenize the cleaned text using NLTK
    tokens = word_tokenize(cleaned_text)
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Function to preprocess data
def preprocess_data(input_path, output_path):
    # Read raw data
    with open(input_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    # Clean and tokenize the text data
    cleaned_text = clean_and_tokenize(raw_text)

    # Save preprocessed data
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

# Function to train MindsDB model
def train_mindsdb_model(input_data, target_column):
    # Load preprocessed data into a Pandas DataFrame
    data = pd.read_csv(input_data)

    # Initialize MindsDB Predictor
    mdb_predictor = Predictor(name='ayurvedic_predictor')

    # Train the MindsDB model
    mdb_predictor.learn(
        from_data=data,
        to_predict=target_column
    )

    # Save MindsDB model
    mdb_predictor.save(MINDSDB_MODEL_DIR)

if __name__ == "__main__":
    # Create the preprocessed data directory if it doesn't exist
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    # Preprocess each Ayurvedic book and save the cleaned data
    for filename in os.listdir(DATA_DIR):
        input_path = os.path.join(DATA_DIR, filename)
        output_path = os.path.join(PREPROCESSED_DIR, f'{os.path.splitext(filename)[0]}_cleaned.txt')
        preprocess_data(input_path, output_path)
        print(f'Preprocessed {filename} and saved to {output_path}')

    # Train the MindsDB model using the preprocessed data
    train_mindsdb_model(input_data='path_to_your_preprocessed_data.csv', target_column='target_column_name')
    print('MindsDB model trained and saved.')
