from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model
from gensim.models import Word2Vec
from utils.text_processing import clean_text
from utils.document_embedding import embed_document
import webbrowser

# Load Keras model
model = load_model('./models/legal_outcome_predictor.keras')

# Load Word2Vec Model
embeddings = Word2Vec.load('./models/word2vec_embeddings.model')

# Initialize Flask application
app = Flask(__name__)

# Webpage application route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        # Retrieve data from JSON
        data = request.get_json(force=True)
    else:
        # Retrieve data from form
        data = request.form

    # Retrieve case facts from data
    case_facts = data.get('case_facts', '')

    # Clean, tokenize, and embed case facts
    case_facts = clean_text(case_facts)
    case_facts_embedding = embed_document(case_facts, embeddings)

    # Get model prediction
    prediction = model.predict(case_facts_embedding.reshape(1, -1))

    if prediction[0][0] > 0.5:
        prediction_label = 'win'
    else:
        prediction_label = 'lose'

    if request.is_json:
        # Return JSON response
        return jsonify({'prediction': prediction_label,
                        'probability': str(prediction[0][0])})
    else:
        # Return webpage
        return render_template('index.html', prediction=prediction_label, probability=str(prediction[0][0]))


if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:5000/')
    app.run(debug=True)