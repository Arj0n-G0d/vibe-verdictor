# Import the PreprocessDatasets function from preprocessing/Preprocess.py 
from preprocessing.Preprocess import PreprocessDatasets  
# Import TfidfVectorizer function from sklearn.feature_extraction.text module for creating a text vectorizer object
from sklearn.feature_extraction.text import TfidfVectorizer 
# Import the Support Vector Machine classifier from sklearn
from sklearn import svm  
# Import the pickle module for saving and loading vectorizers and models 
import pickle  

# SaveModel() makes pickle of the Vectorizer and Model passed to it
def SaveModel(Vectorizer, Model):
    # Open a file named "pickle/Vectorizer.pkl" in binary write mode
    with open("pickle/Vectorizer.pkl", "wb") as file :
        # Serialize and save the Vectorizer to the file
        pickle.dump(Vectorizer, file)
    # Open a file named "pickle/Model.pkl" in binary write mode
    with open("pickle/Model.pkl", "wb") as file:
        # Serialize and save the Model to the file
        pickle.dump(Model, file)

# LoadModel() loads the Vectorizer present in file "pickle/Vectorizer.pkl" and the Model present in file "pickle/Model.pkl" and returns them
def LoadModel():
    # Open the file "pickle/Vectorizer.pkl" in binary read mode
    with open("pickle/Vectorizer.pkl", "rb") as file :
        # Deserialize and load the vectorizer from the file
        vectorizer = pickle.load(file)
    # Open the file "model.pkl" in binary read mode
    with open("pickle/Model.pkl", "rb") as file:
        # Deserialize and load the model from the file
        model = pickle.load(file)
    # Return the loaded vectorizer model 
    return vectorizer, model

# InitModel() creates a TF-IDF vectorizer and a Linear SVC model and fits it on the datasets present at path DatasetPath. Further, It saves the Vectorizer and Model as well as returns them 
def InitModel(DatasetPath):
    # Create a Support Vector Machine model with a linear kernel and C=1.0
    model = svm.SVC(kernel="linear", C=1.0)
    # Preprocess the dataset located at DatasetPath
    dataset = PreprocessDatasets(DatasetPath)
    # Extract the 'text' column from the dataset
    text = dataset.pop("text")
    # Create an instance of the TfidfVectorizer
    vectorizer = TfidfVectorizer()
    # Fit and transform the text data using the vectorizer
    vectorized_text = vectorizer.fit_transform(text)
    # Extract the 'sentiment' column from the dataset
    sentiment = dataset.pop("sentiment")
    # Train the model using the vectorized text data and sentiment labels
    model.fit(vectorized_text, sentiment)
    # Save the fitted Model as well as the fitted Vectorizer
    SaveModel(vectorizer, model)
    # Return the fitted Model as well as the fitted Vectorizer
    return vectorizer, model

# Predict() predicts the sentiment of String using the Vectorizer and Model passed to it
# NOTE : Both Vectorizer and Model both should be fitted on some datasets before making a prediction
def Predict(Vectorizer, Model, String):
    # Create a vector of input String 
    string_vector = Vectorizer.transform([String])
    # Use the model to predict the sentiment of the input string
    return Model.predict(string_vector)[0]
