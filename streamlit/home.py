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
