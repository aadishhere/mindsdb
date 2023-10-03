import spacy
import pandas as pd
from mindsdb import Predictor

# Load the spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Constants
MINDSDB_MODEL_DIR = 'models/mindsdb_model/'

# Function to process text using spaCy and extract entities
def process_text_with_nlp(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]  # Extract named entities from the text
    return entities

# Function to use MindsDB model for prediction
def predict_with_mindsdb(features):
    # Load the MindsDB model
    mdb_predictor = Predictor(name='ayurvedic_predictor')

    # Use the model to predict based on the extracted features
    prediction = mdb_predictor.predict(when=features)
    return prediction[0]['entity_to_predict']

if __name__ == "__main__":
    # Load your preprocessed text data into a list
    texts = ["text1", "text2", ..., "textN"]  # Load your preprocessed text data here

    # Extract entities using spaCy and create a DataFrame
    entities_dataframe = pd.DataFrame(columns=["entity_1", "entity_2"])  # Initialize an empty DataFrame
    for text in texts:
        entities = process_text_with_nlp(text)
        entities_dataframe = entities_dataframe.append(pd.Series(entities, index=entities_dataframe.columns), ignore_index=True)

    # Use MindsDB model for prediction based on the extracted features
    predictions = []
    for index, row in entities_dataframe.iterrows():
        features = {"entity_1": row["entity_1"], "entity_2": row["entity_2"]}  # Replace with actual column names
        prediction = predict_with_mindsdb(features)
        predictions.append(prediction)

    # Combine predictions with original texts (or any other relevant information)
    result_dataframe = pd.DataFrame({"Text": texts, "Prediction": predictions})

    # Save the combined features and predictions as a CSV file
    result_dataframe.to_csv("predictions.csv", index=False)

    print("Entities extracted and predictions saved.")
