import spacy
from mindsdb import Predictor

# Load the spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize MindsDB Predictor
mdb_predictor = Predictor(name='ayurvedic_predictor')

# Function to process user query using spaCy and extract entities
def process_user_query(query):
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]  # Extract named entities from the query
    return entities

# Function to use MindsDB model for prediction
def predict_with_mindsdb(features):
    # Use the MindsDB model to predict based on the extracted features
    prediction = mdb_predictor.predict(when=features)
    return prediction[0]['entity_to_predict']

if __name__ == "__main__":
    print("Ayurvedic Query Processor: Enter your Ayurvedic-related query. Type 'exit' to quit.")
    
    while True:
        # User input
        user_query = input("You: ")

        # Check for exit command
        if user_query.lower() == 'exit':
            print("Exiting. Goodbye!")
            break

        # Extract features from the user query
        features = {"entity_1": None, "entity_2": None}  # Initialize features with None
        extracted_entities = process_user_query(user_query)
        if extracted_entities:
            features["entity_1"] = extracted_entities[0]
            if len(extracted_entities) > 1:
                features["entity_2"] = extracted_entities[1]

        # Use MindsDB model for prediction based on the extracted features
        prediction = predict_with_mindsdb(features)

        # Display the prediction to the user
        if prediction:
            print(f"System: {prediction}")
        else:
            print("System: I'm sorry, I couldn't understand your query. Please try again.")
