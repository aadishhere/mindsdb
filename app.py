from flask import Flask, render_template, request
import tensorflow as tf
import json
import os
from utils.data_preprocessing import preprocess_input_text

app = Flask(__name__)

# Load pre-trained RNN model
model_path = os.path.join("models", "rnn_model.h5")
model = tf.keras.models.load_model(model_path)

# Load processed data (symptoms and remedies)
data_path = os.path.join("data", "processed_data.json")
with open(data_path, "r") as file:
    processed_data = json.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    user_input = request.form['symptoms']
    # Preprocess user input
    processed_input = preprocess_input_text(user_input)
    # Predict remedy using the RNN model
    prediction = model.predict(processed_input)
    # Get the remedy recommendation
    remedy_index = tf.argmax(prediction, axis=1).numpy()[0]
    recommended_remedy = processed_data['remedies'][remedy_index]
    return render_template('result.html', symptoms=user_input, remedy=recommended_remedy)

if __name__ == '__main__':
    app.run(debug=True)
