# Import the Streamlit library
import streamlit as st
# Import the Model class from model/Model.py
from model.Model import Model

# Create two columns in the app layout, with a ratio of 1:9
column1, column2 = st.columns([1, 9])
# Display an image in the first column
column1.image("content/logo.png")
# Display a heading in the second column
column2.markdown("<h2 style='text-align:left; color:black;'>Vibe Verdictor</h2>", unsafe_allow_html=True)

# Create a text input box for user input
text = st.text_input("This app will make judgments about your vibes!! :)")

# Check if the text input is not empty
if text != "" :
    # Create an instance of the Model class
    model = Model()
    # Use the model to predict the vibes based on the user input
    prediction = model.Predict(text)
    # Display the prediction
    st.markdown(f"You have been verdicted of having {prediction} vibes!!")

# Display a footer message
st.markdown("Made with :heart: by Arj0n.G0d")
