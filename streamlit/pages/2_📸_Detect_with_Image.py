import cv2
import streamlit as st
import numpy as np
import os
import json
import time
from dotenv import load_dotenv
import os
from aiobotocore.session import get_session
import asyncio
from PIL import Image

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

# .env 파일 로드
load_dotenv(dotenv_path=".env")

# 언어 초기화 및 선택
init_language()
set_language()
current_language = st.session_state["language"]
text = translations[current_language]["image"]


# S3에 이미지 업로드 (비동기)
async def upload_image_to_s3_async(bucket_name, key, image):
    _, img_encoded = cv2.imencode(".jpg", image)
    session = get_session()
    async with session.create_client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    ) as s3:
        await s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=img_encoded.tobytes(),
            ContentType="image/jpeg",
        )
    return f"https://{bucket_name}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{key}"


# 총 최종 결과 JSON 형식으로 S3에 저장
async def upload_results_to_s3_async(bucket_name, key, data):
    session = get_session()
    async with session.create_client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    ) as s3:
        await s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=json.dumps(data),
            ContentType="application/json",
        )


# 이미지 전처리 함수
def preprocess_image(image):
    processed_image = resize_and_pad_image(
        crop_image(
            apply_color_jitter(image, brightness=1.0, contrast=1.0), crop_ratio=1.0
        )
    )
    return processed_image


# 이미지 결과 표시 및 S3 업로드 함수
def display_results_and_save(images, results, final_result_name):
    st.subheader(text["predict"])

    bucket_name = "cv-7-video"  # S3 버킷 이름
    ng_images = []
    ng_No = []
    ok_No = []
    data = {"final_result_name": final_result_name, "ng_parts": [], "ok_parts": []}

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

    # 최종 결과 표시
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

    # 비동기 s3 업로드 작업
    async def perform_s3_uploads():
        upload_tasks = []

        for idx, img in enumerate(ng_images):
            key = f"results/image/{final_result_name}/NG_part_{idx + 1}.jpg"
            upload_tasks.append(upload_image_to_s3_async(bucket_name, key, img))

        for idx, img in enumerate(ok_No):
            key = f"results/image/{final_result_name}/OK_part_{idx + 1}.jpg"
            upload_tasks.append(upload_image_to_s3_async(bucket_name, key, img))

        # JSON 데이터 업로드 작업 추가
        json_key = f"results/image/{final_result_name}/results.json"
        upload_tasks.append(upload_results_to_s3_async(bucket_name, json_key, data))

        # 모든 작업 비동기 실행
        await asyncio.gather(*upload_tasks)

    st.write("")
    st.write("Upload to S3")
    # 비동기 작업 실행
    with st.spinner("Upload to S3..."):
        asyncio.run(perform_s3_uploads())
        st.success("Complete Upload to S3!")


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

        # 결과 출력 및 S3 저장
        st.success(text["success_processing"])
        current_upload_time = time.strftime("%Y%m%d_%H%M%S")  # 현재 업로드 시간
        final_result_name = f"{current_upload_time}_image_inference"  # 고유 비디오 이름
        display_results_and_save(images, results, final_result_name)


# 프로그램 실행
if __name__ == "__main__":
    image_inference()
