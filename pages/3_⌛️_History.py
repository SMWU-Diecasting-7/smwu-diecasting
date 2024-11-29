import streamlit as st
from translations import init_language, set_language, translations

st.set_page_config(
    page_title="History",
    page_icon="⌛️",
)

# 언어 초기화 및 선택
init_language()
set_language()
current_language = st.session_state["language"]
text = translations[current_language]["history"]

# history 페이지 내용
st.title(text["title"])
st.subheader(text["description"])
st.write(text["select_history"])
