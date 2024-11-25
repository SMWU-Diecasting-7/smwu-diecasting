import boto3
import cv2
import streamlit as st
import numpy as np
import json
from dotenv import load_dotenv
import os
from PIL import Image

# .env 파일 로드
load_dotenv()

# 이미지 해시 함수
def get_image_hash(image):
    resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    avg = gray.mean()
    return ''.join('1' if pixel > avg else '0' for row in gray for pixel in row)

# 해밍 거리 계산 함수
def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# 이미지 전처리 - resize 및 padding
def resize_and_pad_image(img, target_size=(240, 240)):
    if isinstance(img, Image.Image):
        img = np.array(img)
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))
    
    top_pad = (target_size[1] -  new_h) // 2
    bottom_pad = target_size[1] - new_h - top_pad
    left_pad = (target_size[0] - new_w) // 2
    right_pad = target_size[0] - new_w - left_pad
    
    padded_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img.astype(np.uint8)

# 이미지 전처리 - 이미지 밝기 및 대비 조정
def apply_color_jitter(img, brightness=1.2, contrast=1.2):
    lut = np.arange(256, dtype=np.float32)
    lut = lut * contrast + brightness * 50
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # LUT 이미지 변환
    return cv2.LUT(img, lut)

# 이미지 전처리 - Crop 함수 정의
def crop_image(img, crop_ratio):
    h, w = img.shape[:2]
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]

# SageMaker 호출 함수
def invoke_sagemaker_endpoint(endpoint_name, image):
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")
    runtime = boto3.client("sagemaker-runtime",
                           aws_access_key_id=aws_access_key,
                           aws_secret_access_key=aws_secret_key,
                           region_name=aws_region)
    _, img_encoded = cv2.imencode(".jpg", image)
    payload = img_encoded.tobytes()
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-image",
        Body=payload
    )
    result = json.loads(response["Body"].read().decode())
    return result["predicted_class"]

# 이미지 테두리 추가
def add_border(image, color, border_thickness= 50):
    return cv2.copyMakeBorder(
        image,
        top=border_thickness,
        bottom=border_thickness,
        left=border_thickness,
        right=border_thickness,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )