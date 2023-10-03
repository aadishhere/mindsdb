import pandas as pd
from mindsdb import Predictor

# Constants
PREPROCESSED_DATA_FILE = 'path_to_your_preprocessed_data.csv'  # Replace with the actual path to your preprocessed data
TARGET_COLUMN = 'target_column_name'  # Replace with your actual target column name
MINDSDB_MODEL_DIR = 'models/mindsdb_model/'

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
    # Train the MindsDB model using the preprocessed data
    train_mindsdb_model(input_data=PREPROCESSED_DATA_FILE, target_column=TARGET_COLUMN)
    print('MindsDB model trained and saved.')
