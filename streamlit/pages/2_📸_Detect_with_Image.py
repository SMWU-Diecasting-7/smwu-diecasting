import cv2
import streamlit as st
import numpy as np
from translations import init_language, set_language, translations
from utils import (
    resize_and_pad_image,
    crop_image,
    apply_color_jitter,
    invoke_sagemaker_endpoint,
    add_border,
)

st.set_page_config(
    page_title="Detect with Image",
    page_icon="📸",
)

# 언어 초기화 및 선택
init_language()
set_language()
current_language = st.session_state["language"]
text = translations[current_language]["image"]


# 이미지 전처리 함수
def preprocess_image(image):
    # opencv 이미지 전처리
    processed_image = resize_and_pad_image(
        crop_image(
            apply_color_jitter(image, brightness=1.0, contrast=1.0), crop_ratio=1.0
        )
    )

    return processed_image


# 이미지 결과 표시 함수
def display_results(images, results):
    st.subheader(text["predict"])

    ng_images = []
    ng_No = []
    ok_No = []

    # 이미지별 결과 처리
    cols = st.columns(5)
    for i, (image, status) in enumerate(zip(images, results)):
        label = "OK" if status == 1 else "NG"
        label_color = (0, 255, 0) if status == 1 else (0, 0, 255)

        # 이미지 상태 기록
        if status == 0:
            ng_No.append(i + 1)
            ng_images.append(image)
        else:
            ok_No.append(i + 1)

        # 각 이미지별 결과 표시
        bordered_image = add_border(image, label_color)
        cols[i % 5].image(
            bordered_image, channels="BGR", caption=f"Part. {i + 1}: {label}"
        )

    # NG 이미지 추가 출력
    if ng_images:
        st.subheader(text["final_ng"])
        cols = st.columns(5)
        for idx, (ng_image, ng_no) in enumerate(zip(ng_images, ng_No)):
            bordered_ng_image = add_border(ng_image, (0, 0, 255))
            cols[idx % 5].image(
                bordered_ng_image, channels="BGR", caption=f"No. {ng_no}"
            )

    # 최종 결과
    st.subheader(text["summary"])
    if ng_No:
        if current_language == "en":
            st.error(
                f"NG {text['parts']}: {', '.join(map(str, ng_No))} ({text['total']}: {len(ng_No)})"
            )
        elif current_language == "kr":
            st.error(
                f"NG {text['parts']}: {', '.join(map(str, ng_No))} ({text['total']} {len(ng_No)} 개)"
            )

    if ok_No:
        if current_language == "en":
            st.success(
                f"OK {text['parts']}: {', '.join(map(str, ok_No))} ({text['total']}: {len(ok_No)})"
            )
        elif current_language == "kr":
            st.success(
                f"OK {text['parts']}: {', '.join(map(str, ok_No))} ({text['total']} {len(ok_No)} 개)"
            )


def image_inference():
    st.title(text["title"])

    # 이미지 파일 업로드
    uploaded_images = st.file_uploader(
        text["upload"], type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_images:
        st.success(text["upload_success"])
        images = []
        st.subheader(text["uploaded_image"])
        cols = st.columns(len(uploaded_images))

        for idx, uploaded_image in enumerate(uploaded_images):
            # 업로드된 이미지를 읽기
            image = cv2.imdecode(
                np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR
            )
            cols[idx].image(image, channels="BGR", caption=f"Image {idx + 1}")
            images.append(image)

        # 이미지 전처리
        with st.spinner(text["processing"]):
            processed_images = [preprocess_image(img) for img in images]
        # SageMaker 추론
        with st.spinner(text["processing"]):
            results = [
                invoke_sagemaker_endpoint("diecasting-model-T7-endpoint", img)
                for img in processed_images
            ]
        # 결과 출력
        st.success(text["success_processing"])
        display_results(images, results)


# 프로그램 실행
if __name__ == "__main__":
    image_inference()
