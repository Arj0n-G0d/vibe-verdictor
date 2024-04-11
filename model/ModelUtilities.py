# Import the PreprocessDatasets function from preprocessing/Preprocess.py 
from preprocessing.Preprocess import PreprocessDatasets  
# Import the TFIDFVectorizer function from model/Vectorizer.py 
from model.Vectorizer import TFIDFVectorizer  
# Import the Support Vector Machine classifier from sklearn
from sklearn import svm  
# Import the pickle module for saving and loading models
import pickle  

# TrainModel() creates a Linear SVM model and fits it on the datasets present at path DatasetPath 
def TrainModel(DatasetPath):
    # Create a Support Vector Machine classifier with a linear kernel and C=1.0
    classifier = svm.SVC(kernel="linear", C=1.0)
    # Preprocess the dataset located at DatasetPath
    dataset = PreprocessDatasets(DatasetPath)
    # Extract the 'text' column from the dataset
    text = dataset.pop("text")
    # Create an instance of the TFIDFVectorizer
    vectorizer = TFIDFVectorizer()
    # Fit and transform the text data using the vectorizer
    vectorized_text = vectorizer.fit_transform(text)
    # Extract the 'sentiment' column from the dataset
    sentiment = dataset.pop("sentiment")
    # Train the classifier using the vectorized text data and sentiment labels
    classifier.fit(vectorized_text, sentiment)
    # Return the trained classifier
    return classifier

# SaveModel() makes pickle of the Model passed to it
def SaveModel(Model):
    # Open a file named "Model.pkl" in binary write mode
    with open("Model.pkl", "wb") as file:
        # Serialize and save the model to the file
        pickle.dump(Model, file)

# LoadModel() loads the model present in file "Model.pkl" and returns it
def LoadModel():
    # Open the file "Model.pkl" in binary read mode
    with open("Model.pkl", "rb") as file:
        # Deserialize and load the model from the file
        model = pickle.load(file)
    # Return the loaded model
    return model

# Predict() predicts the sentiment of String using the Model passed to it
def Predict(Model, String):
    # Use the model to predict the sentiment of the input string
    return Model.predict([String])[0]
