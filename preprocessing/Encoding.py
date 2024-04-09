import chardet  # Import the chardet library for character encoding detection

# DetectEncoding() attempts to detect the encoding of a file located at the given FilePath
def DetectEncoding(FilePath) :  
    with open(FilePath,"rb") as file :  # Open the file in binary mode to read bytes
        detector = chardet.universaldetector.UniversalDetector()  # Create a UniversalDetector object from chardet
        for line in file :  # Iterate over each line in the file
            detector.feed(line)  # Feed the line to the detector to analyze its encoding
            if(detector.done) :  # If the detector has finished analyzing the encoding, exit the loop
                break
        detector.close()  # Close the detector to finalize the encoding detection
    return detector.result["encoding"]  # Return the detected encoding from the detector's result