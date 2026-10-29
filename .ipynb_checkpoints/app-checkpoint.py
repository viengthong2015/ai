import streamlit as st

st.title("My first Streamlit app")

name = st.text_input("Enter your name")
if name:
    st.success(f"Hello, {name}!")
