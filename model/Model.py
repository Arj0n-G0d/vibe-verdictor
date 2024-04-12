# Import the PreprocessDatasets function from preprocessing/Preprocess.py 
from preprocessing.Preprocess import PreprocessDatasets  
# Import TfidfVectorizer function from sklearn.feature_extraction.text module for creating a text vectorizer object
from sklearn.feature_extraction.text import TfidfVectorizer 
# Import train_test_split function from sklearn.model_selection module for splitting the dataset
from sklearn.model_selection import train_test_split
# Import the Support Vector Machine classifier from sklearn
from sklearn import svm 
# Import the pickle module for saving and loading models
import pickle  

# SaveModel() makes pickle of the Model passed to it
def SaveModel(Model) :
    # Open a file named "pickle/Model.pkl" in binary write mode
    with open("pickle/Model.pkl", "wb") as file:
        # Serialize and save the Model to the file
        pickle.dump(Model, file)

# LoadModel() loads the Model present in file "pickle/Model.pkl" and returns it's data members
def LoadModel() :
    # Open the file "pickle/Model.pkl" in binary read mode
    with open("pickle/Model.pkl", "rb") as file:
        # Deserialize and load the model from the file
        model = pickle.load(file)
    # Return the loaded model's data members
    return model.Model, model.Vectorizer, model.Accuracy

class Model :
    def __init__(self, ModelType="pretrained", DatasetPath="dataset"):
        # Load a pretrained model if ModelType is "pretrained"
        if ModelType == "pretrained":
            self.Model, self.Vectorizer, self.Accuracy = LoadModel()
        # Train a new model if ModelType is "train"
        elif ModelType == "train":
            # Create a Support Vector Machine model with a linear kernel and C=1.0
            self.Model = svm.SVC(kernel="linear", C=1.0)
            # Create a TF-IDF vectorizer
            self.Vectorizer = TfidfVectorizer()
            # Preprocess the dataset located at DatasetPath
            data = PreprocessDatasets(DatasetPath)
            # Split the dataset into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(data["text"], data["sentiment"])
            # Fit and transform the training text data using the vectorizer
            x_train_transformed = self.Vectorizer.fit_transform(x_train)
            # Transform the testing text data using the vectorizer
            x_test_transformed = self.Vectorizer.transform(x_test)
            # Train the model using the vectorized training text data and sentiment labels
            self.Model.fit(x_train_transformed, y_train)
            # Calculate the accuracy of the model on the testing set
            self.Accuracy = self.Model.score(x_test_transformed, y_test)
            # Save the trained model and vectorizer
            SaveModel(self)
        else:
            # Raise an exception if an invalid ModelType is provided
            raise Exception(f"{ModelType} is NOT a valid ModelType!!")

    def __str__(self):
        # Return a string representation of the model
        return "LinearSVC Model"

    def Predict(self, String):
        # Transform the input string using the vectorizer
        vectorized_string = self.Vectorizer.transform([String])
        # Predict the sentiment of the input string using the model
        return self.Model.predict(vectorized_string)[0]