import streamlit as st


def init_language():
    # 세션 초기화
    if "language" not in st.session_state:
        st.session_state["language"] = "en"  # 기본언어 영어로 설정


def set_language():
    # 언어 선택 라디오 버튼
    language_selection = selected_language = st.sidebar.radio(
        "🌐 Select Language",
        ["🇺🇸 ENGILSH", "🇰🇷 한국어"],
    )

    # 세션 업데이트
    if language_selection == "🇺🇸 ENGILSH":
        st.session_state["language"] = "en"
    else:
        st.session_state["language"] = "kr"


translations = {
    "en": {
        "home": {
            "title": "Welcome to Sookmyung Diecasting",
            "subtitle": "SageMaker-powered diecasting quality classification system.",
            "description": "Use the sidebar to analyze videos, images, or review past inference results.",
        },
        "video": {
            "title": "Realtime Video Analysis",
            "upload": "Upload a video file for analysis.",
            "upload_success": "File upload completed!",
            "processing": "Analyzing video...",
            "summary": "Final Result Summary",
            "total": "Total",
            "parts": "Parts",
            "detailed_image": "Detailed Images",
            "select_img_box": "Select Part to View Images",
        },
        "image": {
            "title": "Realtime Image Analysis",
            "upload": "Upload an image file for analysis.",
            "upload_success": "File upload completed!",
            "uploaded_image": "Uploaded Image",
            "processing": "Analyzing image...",
            "success_processing": "Inference completed!",
            "total": "Total",
            "parts": "Parts",
            "result": "Result",
            "predict": "Predict Result",
            "final_ng": "Final NG Parts",
            "summary": "Final Result Summary",
            "ng_part": "NG Parts",
            "ok_part": "OK Parts",
        },
        "history": {
            "title": "Inference History",
            "description": "Review past video and image analysis results.",
            "select_history": "Select a record to view details.",
        },
    },
    "kr": {
        "home": {
            "title": "숙명 다이캐스팅에 오신 것을 환영합니다.",
            "subtitle": "SageMaker 기반 다이캐스팅 품질 분류 시스템",
            "description": "사이드바를 사용하여 비디오, 이미지 분석 또는 이전 분석 결과를 확인하세요.",
        },
        "video": {
            "title": "실시간 비디오 분석",
            "upload": "분석할 비디오 파일을 업로드하세요.",
            "upload_success": "파일 업로드가 완료되었습니다!",
            "processing": "비디오를 분석 중입니다...",
            "summary": "최종 결과 요약",
            "total": "총",
            "parts": "부품",
            "detailed_image": "상세 이미지",
            "select_img_box": "이미지를 볼 부품을 선택하세요",
        },
        "image": {
            "title": "실시간 이미지 분석",
            "upload": "분석할 이미지 파일을 업로드하세요.",
            "upload_success": "파일 업로드가 완료되었습니다!",
            "uploaded_image": "업로드된 이미지",
            "processing": "이미지를 분석 중입니다...",
            "success_processing": "분석이 완료되었습니다!",
            "total": "총",
            "parts": "부품",
            "result": "결과",
            "predict": "예측 결과",
            "final_ng": "최종 NG 부품",
            "summary": "최종 결과 요약",
            "ng_part": "NG 부품",
            "ok_part": "OK 부품",
        },
        "history": {
            "title": "분석 기록",
            "description": "이전에 분석된 비디오와 이미지 결과를 확인하세요.",
            "select_history": "세부 정보를 볼 기록을 선택하세요.",
        },
    },
}
