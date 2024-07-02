import numpy as np
from gensim.models import Word2Vec

# Define a function to generate a document embedding when given a tokenized document
def embed_document(document, model):
    # Initialize empty string of word vectors
    word_vectors = []
    
    # Get valid words    
    for word in document:
        if word in model.wv.index_to_key:
            word_vectors.append(model.wv[word])
    
    # Return zero vector if list is empty 
    if not word_vectors:
        return np.zeros(model.vector_size)

    # Get document embedding by averaging word vectors
    document_embedding = np.mean(word_vectors, axis=0)
    
    return document_embedding