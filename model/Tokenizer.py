# Import the SpaCy library for NLP processing
import spacy
# Import the string module for working with string operations
import string

# SpacyTokenizer() tokenizes a string using SpaCy
def SpacyTokenizer(String):
    # Load the English tokenizer, tagger, parser, NER, and word vectors
    nlp = spacy.load("en_core_web_sm")
    # Process the input string using the loaded NLP pipeline
    doc = nlp(String)
    # Get the default list of stop words from the English language
    stop_words = nlp.Defaults.stop_words
    # Get the set of punctuation characters
    punctuation = string.punctuation
    # Lemmatize each word in the document and convert it to lowercase
    tokens = [word.lemma_.lower() for word in doc]
    # Filter out tokens that are punctuation or stop words
    tokens = [word for word in tokens if word not in punctuation and word not in stop_words]
    # Return the list of tokens
    return tokens