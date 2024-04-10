# Import the os module for interacting with the operating system
import os  
# Import the pandas library for data manipulation
import pandas as pd  
# Import the DetectEncoding function from preprocessing/Encoding.py
from preprocessing.Encoding import DetectEncoding  

# PreprocessDatasets() preprocesses all the datasets present in FolderPath into a single dataset and also performs other necessary preprocessing on it
# NOTE : Each dataset must contain "text" and "sentiment" columns
def PreprocessDatasets(FolderPath) :  
    # Create an empty list to store the datasets
    datasets = []  
    # Iterate over each file in the specified folder
    for dataset_name in os.listdir(FolderPath) :  
        # Create the full path to the dataset file
        dataset_path = FolderPath+"/"+dataset_name  
        # Read the dataset using pandas, specifying the columns 'text' and 'sentiment', and using the detected encoding
        ds = pd.read_csv(dataset_path, usecols=["text", "sentiment"], encoding=DetectEncoding(dataset_path))
        # Append the dataset to the list of datasets
        datasets.append(ds)  
    # Concatenate all datasets into a single dataframe
    final_dataset = pd.concat(datasets)  

    # Reset the index of the final dataset and modify it in place
    final_dataset.reset_index(drop=True, inplace=True)  
    # Drop any rows with missing values from the final dataset
    final_dataset.dropna(inplace=True)  
    # Remove the 'sentiment' column from the final dataset and store it separately
    sentiment = final_dataset.pop("sentiment") 
    # Converting all strings in sentiment to lowercase
    sentiment.str.lower() 
    # Insert the modified 'sentiment' column back into the final dataset
    final_dataset.insert(1, "sentiment", sentiment)  
    # Return the processed final dataset
    return final_dataset  