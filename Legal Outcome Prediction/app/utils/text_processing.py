import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define a function to clean and tokenize a text document
def clean_text(text):
    # Remove non-word characters
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    
    # Tokenize sentence
    tokens = word_tokenize(text)
    
    # Initialize empty list of cleaned tokens
    cleanTokens = []

    # Define additional stopwords used in pre-processing
    new_stopwords = ['plaintiff', 'class', 'defendant', 'c', 'defendants', 'u',
                    'action', 'b', 'law', 'plaintiffs', '1', '2', 'herein', 'behalf']
    
    # Remove common English stop-words
    for word in tokens:
        if word not in stopwords.words('english') and word not in new_stopwords:
            cleanTokens.append(word)
    
    return cleanTokens