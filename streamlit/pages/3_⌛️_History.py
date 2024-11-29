import streamlit as st
from translations import init_language, set_language, translations
import boto3
import json
from PIL import Image
import os 
from io import BytesIO
from utils import (
    get_image_hash,
    hamming_distance,
    resize_and_pad_image,
    crop_image,
    apply_color_jitter,
    add_border,
    invoke_sagemaker_endpoint,
)

st.set_page_config(
    page_title="History",
    page_icon="⌛️",
)



# S3 클라이언트 생성 함수
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )


# JSON 데이터를 S3에서 가져오기
def fetch_json_from_s3(bucket_name, json_key):
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket_name, Key=json_key)
    return json.loads(response["Body"].read().decode("utf-8"))


# 이미지를 S3에서 가져오기
def fetch_image_from_s3(bucket_name, image_key):
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket_name, Key=image_key)
    return Image.open(BytesIO(response["Body"].read()))

def deduplicate_parts(parts):
    unique_parts = []
    seen_part_numbers = set()

    for part in parts:
        # part가 리스트인지 확인 (유효성 검사)
        if isinstance(part, list) and len(part) > 0 and isinstance(part[0], dict):
            part_number = part[0]["part_number"]
            if part_number not in seen_part_numbers:
                unique_parts.append(part)
                seen_part_numbers.add(part_number)
        else:
            st.warning(f"잘못된 데이터 형식: {part}")
    
    return unique_parts


# NG Part 렌더링
@st.fragment
def render_ng_parts(result_data, bucket_name):
    ng_part_numbers = [part[0]["part_number"] for part in result_data["ng_parts"]]
    selected_ng_part = st.selectbox(
        "Choose an NG part:",
        ["Select data"] + sorted(ng_part_numbers),
        key="selected_ng_part",
    )

    if selected_ng_part != "Select data":
        st.subheader(f"NG Part {selected_ng_part}")
        for part in result_data["ng_parts"]:
            if part[0]["part_number"] == selected_ng_part:
                cols = st.columns(5)  # 5개씩 출력
                for idx, img_info in enumerate(part):
                    image_url = img_info["image_url"]
                    image_key = image_url.replace(f"s3://{bucket_name}/", "")
                    img = fetch_image_from_s3(bucket_name, image_key)
                    cols[idx % 5].image(
                        img, caption=f"Part {selected_ng_part} - Image {idx + 1}"
                    )


# OK Part 렌더링
@st.fragment
def render_ok_parts(result_data, bucket_name):
    ok_part_numbers = [part[0]["part_number"] for part in result_data["ok_parts"]]
    selected_ok_part = st.selectbox(
        "Choose an OK part:",
        ["Select data"] + sorted(ok_part_numbers),
        key="selected_ok_part",
    )

    if selected_ok_part != "Select data":
        st.subheader(f"OK Part {selected_ok_part}")
        for part in result_data["ok_parts"]:
            if part[0]["part_number"] == selected_ok_part:
                cols = st.columns(5)  # 5개씩 출력
                for idx, img_info in enumerate(part):
                    image_url = img_info["image_url"]
                    image_key = image_url.replace(f"s3://{bucket_name}/", "")
                    img = fetch_image_from_s3(bucket_name, image_key)
                    cols[idx % 5].image(
                        img, caption=f"Part {selected_ok_part} - Image {idx + 1}"
                    )


# Streamlit 메인 UI
def view_results_from_s3():
    st.title("View Results from S3")

    # S3 버킷과 경로
    bucket_name = "cv-7-video"
    results_prefix = "results/"

    # JSON 파일 선택
    st.subheader("Select a JSON file")
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=results_prefix)
    json_files = [
        obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".json")
    ]

    if not json_files:
        st.warning("No JSON files found.")
        return

    selected_json = st.selectbox(
        "Choose a JSON file:", ["Select a JSON file"] + json_files, key="selected_json"
    )

    if selected_json == "Select a JSON file":
        st.info("Please select a JSON file to view results.")
        return

    # JSON 데이터 로드 및 중복 제거
    @st.cache_data
    def load_json_data(bucket_name, json_key):
        result_data = fetch_json_from_s3(bucket_name, json_key)
        result_data["ng_parts"] = deduplicate_parts(result_data["ng_parts"])
        result_data["ok_parts"] = deduplicate_parts(result_data["ok_parts"])
        return result_data

    result_data = load_json_data(bucket_name, selected_json)
    st.success(f"Successfully loaded results: {result_data['video_name']}")

    # NG 및 OK Part 독립적 렌더링
    render_ng_parts(result_data, bucket_name)
    render_ok_parts(result_data, bucket_name)


if __name__ == "__main__":
    view_results_from_s3()

# 언어 초기화 및 선택
init_language()
set_language()
current_language = st.session_state["language"]
text = translations[current_language]["history"]

# # history 페이지 내용
# st.title(text["title"])
# st.subheader(text["description"])
# st.write(text["select_history"])