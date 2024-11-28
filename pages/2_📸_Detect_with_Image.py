import cv2
import streamlit as st
import numpy as np
import os
import json
import boto3
import time  # time 모듈 추가
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

# S3 클라이언트 생성
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

# S3에 이미지 업로드
def upload_image_to_s3(bucket_name, key, image):
    _, img_encoded = cv2.imencode(".jpg", image)
    s3 = get_s3_client()
    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=img_encoded.tobytes(),
        ContentType="image/jpeg",
    )
    return f"https://{bucket_name}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{key}"

# JSON 데이터 S3에 저장
def upload_results_to_s3(bucket_name, key, data):
    s3 = get_s3_client()
    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps(data),
        ContentType="application/json",
    )

# 이미지 전처리 함수
def preprocess_image(image):
    processed_image = resize_and_pad_image(
        crop_image(
            apply_color_jitter(image, brightness=1.3, contrast=1.15), crop_ratio=1.0
        )
    )
    return processed_image

# 이미지 결과 표시 및 저장 함수
def display_results_and_save(images, results, video_name):
    st.subheader("Predict Result")

    bucket_name = "cv-7-video"  # S3 버킷 이름
    ng_images = []
    ok_images = []
    ng_parts = []
    ok_parts = []
    data = {"video_name": video_name, "ng_parts": [], "ok_parts": []}

    # 이미지별 결과 처리
    cols = st.columns(5)
    for i, (image, status) in enumerate(zip(images, results)):
        label = "OK" if status == 1 else "NG"
        label_color = (0, 255, 0) if status == 1 else (0, 0, 255)

        # 이미지 상태 기록
        if status == 0:
            ng_images.append(image)
            ng_parts.append(i + 1)
        else:
            ok_images.append(image)
            ok_parts.append(i + 1)

        # 각 이미지별 결과 표시
        bordered_image = add_border(image, label_color)
        cols[i % 5].image(
            bordered_image, channels="BGR", caption=f"Part. {i + 1}: {label}"
        )

    # NG 이미지 업로드
    for idx, img in enumerate(ng_images):
        key = f"results/{video_name}/NG_part_{idx + 1}.jpg"
        image_url = upload_image_to_s3(bucket_name, key, img)
        data["ng_parts"].append({"part_number": idx + 1, "image_url": image_url})

    # OK 이미지 업로드
    for idx, img in enumerate(ok_images):
        key = f"results/{video_name}/OK_part_{idx + 1}.jpg"
        image_url = upload_image_to_s3(bucket_name, key, img)
        data["ok_parts"].append({"part_number": idx + 1, "image_url": image_url})

    # NG 이미지 추가 출력
    if ng_images:
        st.subheader("Final NG Parts")
        cols = st.columns(5)
        for idx, (ng_image, ng_no) in enumerate(zip(ng_images, ng_parts)):
            bordered_ng_image = add_border(ng_image, (0, 0, 255))
            cols[idx % 5].image(
                bordered_ng_image, channels="BGR", caption=f"No. {ng_no}"
            )

    # 최종 결과 JSON 저장
    json_key = f"results/{video_name}/results.json"
    upload_results_to_s3(bucket_name, json_key, data)

    # 최종 결과 표시
    st.subheader("Final Result Summary")
    if ng_parts:
        st.error(f"NG Parts: {', '.join(map(str, ng_parts))} (Total: {len(ng_parts)})")
    if ok_parts:
        st.success(f"OK Parts: {', '.join(map(str, ok_parts))} (Total: {len(ok_parts)})")

# 이미지 추론 메인 함수
def image_inference():
    st.title("Real-time NG/OK Image Classification")

    # 이미지 파일 업로드
    uploaded_images = st.file_uploader(
        "Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_images:
        video_name = f"image_inference_{int(time.time())}"  # 고유 비디오 이름
        images = []
        st.subheader("Uploaded Images")
        cols = st.columns(len(uploaded_images))

        for idx, uploaded_image in enumerate(uploaded_images):
            # 업로드된 이미지를 읽기
            image = cv2.imdecode(
                np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR
            )
            cols[idx].image(image, channels="BGR", caption=f"Uploaded Image {idx + 1}")
            images.append(image)

        # 이미지 전처리
        with st.spinner("Processing Images..."):
            processed_images = [preprocess_image(img) for img in images]

        # SageMaker 추론
        with st.spinner("Analyzing Images..."):
            results = [
                invoke_sagemaker_endpoint("diecasting-model-T7-endpoint", img)
                for img in processed_images
            ]

        # 결과 표시 및 저장
        st.success("Inference Complete!")
        display_results_and_save(images, results, video_name)

# 프로그램 실행
if __name__ == "__main__":
    image_inference()
