# Import the PreprocessDatasets function from preprocessing/Preprocess.py 
from preprocessing.Preprocess import PreprocessDatasets  
# Import TfidfVectorizer function from sklear.feature_extraction.text module for creating a text vectorizer object
from sklearn.feature_extraction.text import TfidfVectorizer 
# Import the Support Vector Machine classifier from sklearn
from sklearn import svm  
# Import the pickle module for saving and loading models
import pickle  

# TrainModel() creates a Linear SVC model and fits it on the datasets present at path DatasetPath 
def TrainModel(DatasetPath):
    # Create a Support Vector Machine classifier with a linear kernel and C=1.0
    classifier = svm.SVC(kernel="linear", C=1.0)
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
    # Train the classifier using the vectorized text data and sentiment labels
    classifier.fit(vectorized_text, sentiment)
    # Return the fitted Model as well as the fitted Vectorizer
    return vectorizer, classifier

# SaveModel() makes pickle of the Model passed to it
def SaveModel(Model):
    # Open a file named "model.pkl" in binary write mode
    with open("model.pkl", "wb") as file:
        # Serialize and save the model to the file
        pickle.dump(Model, file)

# LoadModel() loads the model present in file "model.pkl" and returns it
def LoadModel():
    # Open the file "model.pkl" in binary read mode
    with open("model.pkl", "rb") as file:
        # Deserialize and load the model from the file
        model = pickle.load(file)
    # Return the loaded model
    return model

# Predict() predicts the sentiment of String using the Model and Vectorizer passed to it
# NOTE : Both Model and Vectorizer both should be fitted on some datasets before making a prediction
def Predict(Model, Vectorizer, String):
    # Create a vector of input String 
    string_vector = Vectorizer.transform([String])
    # Use the model to predict the sentiment of the input string
    return Model.predict(string_vector)[0]
