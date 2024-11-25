import cv2
import streamlit as st
import numpy as np
from utils import resize_and_pad_image, crop_image, apply_color_jitter, invoke_sagemaker_endpoint, add_border

st.set_page_config(
    page_title="Detect with Image",
    page_icon="📸",
)

# 이미지 전처리 함수
def preprocess_image(image):
    processed_image = resize_and_pad_image(
        crop_image(apply_color_jitter(image, brightness=1.15, contrast=1.15), crop_ratio=0.97)
    )
    return processed_image

# 이미지 결과 표시 함수
def display_results(images, results):
    st.subheader("Predict Result")
    
    ng_images = []
    ng_No = []
    ok_No = []

    # 5개씩 묶어서 처리
    for i in range(0, len(results), 5):
        batch_results = results[i:i+5]
        batch_images = images[i:i+5]
        
        # 부품 상태 결정
        batch_status = "NG" if 0 in batch_results else "OK"

        if batch_status == "NG":
            ng_No.append(i)
        else:
            ok_No.append(i)

        cols = st.columns(5)
        for j, (image, status) in enumerate(zip(batch_images, batch_results)):
            label = "OK" if status == 1 else "NG"
            label_color = (0, 255, 0) if status == 1 else (0, 0, 255)

            if status == 0:
                ng_images.append(image)

            bordered_image = add_border(image, label_color)
            cols[j].image(bordered_image, channels="BGR", caption=f"No. {i + j + 1}: {label}")
    
    # NG 이미지 추가 출력
    if ng_images:
        st.subheader("Final NG Images")
        cols = st.columns(5)
        for idx, ng_image in enumerate(ng_images):
            bordered_ng_image = add_border(ng_image, (0, 0, 255))
            cols[idx % 5].image(bordered_ng_image, channels="BGR", caption=f"No. {idx + 1}")
    
    # 최종 결과
    st.subheader("Final Result Summary")
    if ng_No:
        st.error(f"NG Parts: {', '.join(map(str, ng_No))} (Total: {len(ng_No)})")
    if ok_No:
        st.success(f"OK Parts: {', '.join(map(str, ok_No))} (Total: {len(ok_No)})")


def image_inference():
    st.title("Real-time NG/OK Image Classification")
    
    # 이미지 파일 업로드
    uploaded_images = st.file_uploader("Choose an image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images:
        images = []
        st.subheader("Uploaded Images")
        cols = st.columns(len(uploaded_images))
        
        for idx, uploaded_image in enumerate(uploaded_images):
            # 업로드된 이미지를 읽기
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
            cols[idx].image(image, channels="BGR", caption=f"Uploaded Image {idx + 1}")
            images.append(image)
        
        # 이미지 전처리
        with st.spinner("Processing Images..."):
            processed_images = [preprocess_image(img) for img in images]
        # SageMaker 추론
        with st.spinner("Analyzing Iamges..."):
            results = [invoke_sagemaker_endpoint("test-endpoint", img) for img in processed_images]
        # 결과 출력
        st.success("Inference Complete!")
        display_results(processed_images, results)
            
                

# 프로그램 실행
if __name__ == "__main__":
    image_inference()
