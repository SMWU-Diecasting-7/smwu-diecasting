import streamlit as st
from translations import init_language, set_language, translations

st.set_page_config(
    page_title="Dieasting Classification",
    page_icon=":camera:",
)

# 언어 초기화 및 선택
init_language()
set_language()
current_language = st.session_state["language"]
text = translations[current_language]["home"]

# home 페이지 내용
st.title(text["title"])
st.subheader(text["subtitle"])
st.write(text["description"])
st.write("")

st.subheader("\n\nHow to Analyze with 📹 VIDEO?")
st.write(
    "1. Upload a video file for analysis. Ensure the file size is less than 200MB."
)
st.write("2. Wait for the processing to complete.")
st.write("3. Once the processing is done, a 'Result Summary' will be displayed.")
st.write(
    "4. You can view images of all NG/OK parts. Use the select bar to choose specific part numbers."
)

st.subheader("How to Analyze with 📸 IMAGE?")
st.write("1. Upload an image file for analysis.")
st.write("2. Wait for the processing to complete.")
st.write("3. Once the processing is done, a 'Result Summary' will be displayed.")
st.write("4. You can view images of all NG/OK parts.")