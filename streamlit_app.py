import streamlit as st
from nlp_utils import process_input

st.title("Smart Task Manager AI")
user_input = st.text_input("How can I help you today?")
if user_input:
    response = process_input(user_input)
    st.write(response)