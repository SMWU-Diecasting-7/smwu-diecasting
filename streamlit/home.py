import streamlit as st
from translations import translations

st.set_page_config(
    page_title="Dieasting Classification",
    page_icon=":camera:",
)

if "language" not in st.session_state:
    st.session_state.language = "en"  # 기본언어 영어로 설정

# 언어 선택 라디오 버튼
selected_language = st.sidebar.radio(
    "Select Language",
    options=["en", "kr"],
    format_func=lambda x: "🇺🇸 ENGILISH" if x == "en" else "🇰🇷 한국어",
)

# 사용자 선택 언어 업데이터
if selected_language != st.session_state.language:
    st.session_state.language = selected_language

# 현재 선택된 언어
current_language = st.session_state.language


def home():
    # 현재 선택된 언어
    current_language = st.session_state.language
    text = translations[current_language]["home"]

    st.title(text["title"])
    st.subheader(text["subtitle"])
    st.write(text["description"])


if __name__ == "__main__":
    home()
