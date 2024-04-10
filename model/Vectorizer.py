# Import the SpacyTokenizer function from model/Tokenizer.py
from model.Tokenizer import SpacyTokenizer
# Import the TfidfVectorizer class from the sklearn.feature_extraction.text module
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to create a TF-IDF vectorizer using SpaCy tokenizer
def TFIDFVectorizer():
    # Create an instance of the TfidfVectorizer with the SpaCy tokenizer
    tfidf_vectorizer = TfidfVectorizer(tokenizer=SpacyTokenizer)
    # Return the TF-IDF vectorizer
    return tfidf_vectorizer
