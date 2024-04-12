import streamlit as st
from model.Model import Model
column1, column2 = st.columns([1,9])
column1.image("content/Pasted image 1.png")
column2.markdown("<h2 style = 'text-align:left; color:black;'>Vibe Verdictor</h2>",unsafe_allow_html=True)
text = st.text_input("This app will make judgements about your vibes!! :)")
if(text!="") :
    model = Model()
    prediction = model.Predict(text)
    st.markdown(f"You have been verdicted of having {prediction} vibes!!")
st.markdown("Made with :heart: by this_dot_god")