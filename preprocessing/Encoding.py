# Import the chardet library for character encoding detection
import chardet 

# DetectEncoding() attempts to detect the encoding of a file located at the given FilePath
def DetectEncoding(FilePath) :  
    # Open the file in binary mode to read bytes
    with open(FilePath,"rb") as file :  
        # Create a UniversalDetector object from chardet
        detector = chardet.universaldetector.UniversalDetector()  
        # Iterate over each line in the file
        for line in file :  
            # Feed the line to the detector to analyze its encoding
            detector.feed(line)  
            # If the detector has finished analyzing the encoding, exit the loop
            if(detector.done) :  
                break
        # Close the detector to finalize the encoding detection
        detector.close()  
    # Return the detected encoding from the detector's result
    return detector.result["encoding"]  