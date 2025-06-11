import streamlit as st
import os

st.title("Live People Count & Posture Detection")
st.write("Press button to start live detection.")
if st.button("Start Detection"):
    os.system("python3 main.py")
