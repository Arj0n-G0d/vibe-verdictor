import os  # Import the os module for interacting with the operating system
import pandas as pd  # Import the pandas library for data manipulation
from preprocessing.Encoding import DetectEncoding  # Import the DetectEncoding function from preprocessing/Encoding.py

# PreprocessDatasets() preprocesses all the datasets present in FolderPath into a single dataset and also performs other necessary preprocessing on it
# NOTE : Each dataset must contain "text" and "sentiment" columns
def PreprocessDatasets(FolderPath) :  # Define a function called PreprocessDatasets that takes a FolderPath parameter
    datasets = []  # Create an empty list to store the datasets
    for dataset_name in os.listdir(FolderPath) :  # Iterate over each file in the specified folder
        dataset_path = FolderPath+"/"+dataset_name  # Create the full path to the dataset file
        # Read the dataset using pandas, specifying the columns 'text' and 'sentiment', and using the detected encoding
        ds = pd.read_csv(dataset_path, usecols=["text", "sentiment"], encoding=DetectEncoding(dataset_path))
        datasets.append(ds)  # Append the dataset to the list of datasets
    final_dataset = pd.concat(datasets)  # Concatenate all datasets into a single dataframe

    final_dataset.reset_index(drop=True, inplace=True)  # Reset the index of the final dataset and modify it in place
    final_dataset.dropna(inplace=True)  # Drop any rows with missing values from the final dataset
    sentiment = final_dataset.pop("sentiment")  # Remove the 'sentiment' column from the final dataset and store it separately
    for i in range(len(sentiment)) :  # Iterate over each sentiment value
        sentiment[i] = sentiment[i].lower()  # Convert the sentiment value to lowercase
    final_dataset.insert(1, "sentiment", sentiment)  # Insert the modified 'sentiment' column back into the final dataset
    return final_dataset  # Return the preprocessed final dataset