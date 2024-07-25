# Legal Outcome Prediction

In this project, I build a legal outcome prediction model that predicts the verdict of class action lawsuits in the U.S. based on a plain-text discription of case facts. The project uses natural language processing (NLP) techniques to extract meaning from over 30 million words and a neural network to make predictions. The model is then deployed in a web-based API.

## Results
The legal outcome prediction model yields a validation accuracy of 65.5%. Although the raw accuracy score appears relatively low, the model is 24% more accurate than [human experts](https://huggingface.co/datasets/darrow-ai/USClassActionOutcomes_ExpertsAnnotations) faced with a random sample of the same dataset.

## Project Structure

- **app/**: Contains the Flask application for the model API.
  - `main.py`: The Flask application's main file that handles requests and predictions.
  - `static/`: Contains CSS for the web interface.
  - `templates/`: Contains the HTML template for the web interface.
  - `utils/`: Contains Python functions to clean, pre-process, and embed an input case description.

- **models/**: Contains the saved models.
  - `legal_outcome_predictor.keras`: The trained Keras neural network for predicting case outcomes.
  - `word2vec_model.pkl`: The trained Word2Vec model for generating word embeddings.

- **notebooks/**: Contains Jupyter notebooks for data pre-processing, feature engineering, and model training.
  - `data_pre_processing.ipynb`: Notebook for document pre-processing and tokenization.
  - `feature_engineering.ipynb`: Notebook for generating word and document embeddings.
  - `modeling.ipynb`: Notebook for training and evaluating classification models.

- **data/**: Contains all data files.
  - `cleaned_class_action.json`: Contains cleaned and tokenized case descriptions.
  - `X.npy`: Contains the feature matrix of embedded documents.
  - `y.npy`: Contains the target vector of case outcomes.
  - `modeling.ipynb`: Contains the target vector.

- **requirements.txt**: List of Python dependencies.
  
- **README.md**: Project overview.

- **LICENSE**: Project license.

## Setup Instructions

### Prerequisites

- Python 3.7+
- pip

### Installation

1. **Install dependencies:**
   ```sh
   cd 'Legal Outcome Prediction'
   pip install r requirements.txt
   ```

2. **Run the Flask application:**
   ```sh
   python app/main.py
   ```

## Usage

1. **Input case facts:**
    Input case facts inside the web interface.

2. **Predict outcome:**
    Click the "Predict" button to obtain the case prediction, as well as the prediction probability.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) for details.
