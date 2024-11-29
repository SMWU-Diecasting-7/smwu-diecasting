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

# 중복 제거 로직
def remove_duplicate_parts(data):
    def deduplicate(parts):
        seen = set()
        unique_parts = []
        for part in parts:
            # Serialize the part as a tuple for deduplication
            part_serialized = tuple(tuple(d.items()) for d in part)
            if part_serialized not in seen:
                seen.add(part_serialized)
                unique_parts.append(part)
        return unique_parts

    # Apply deduplication to NG and OK parts
    data["ng_parts"] = deduplicate(data.get("ng_parts", []))
    data["ok_parts"] = deduplicate(data.get("ok_parts", []))
    return data

# Streamlit UI
def view_results_from_s3():
    st.title("S3에서 결과 보기")

    # S3 버킷과 경로
    bucket_name = "cv-7-video"
    results_prefix = "results/"

    # JSON 파일 선택
    st.subheader("결과 JSON 파일 선택")
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=results_prefix)
    json_files = [
        obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".json")
    ]

    if not json_files:
        st.warning("결과 JSON 파일이 없습니다.")
        return

    selected_json = st.selectbox("결과 JSON 파일을 선택하세요:", json_files)

    # 데이터 로드
    if st.button("결과 가져오기"):
        try:
            # JSON 데이터 로드
            result_data = fetch_json_from_s3(bucket_name, selected_json)
            # 중복 제거
            result_data = remove_duplicate_parts(result_data)
            st.success(f"결과를 성공적으로 로드했습니다: {result_data['video_name']}")

            # NG 파트 표시
            st.header("NG Parts")
            for part in result_data["ng_parts"]:
                part_number = part[0]["part_number"]
                st.subheader(f"NG Part {part_number}")
                cols = st.columns(5)  # 5개씩 출력
                for idx, img_info in enumerate(part):
                    image_url = img_info["image_url"]
                    image_key = image_url.replace(f"s3://{bucket_name}/", "")
                    img = fetch_image_from_s3(bucket_name, image_key)
                    cols[idx % 5].image(img, caption=f"Part {part_number} - Image {idx + 1}")

            # OK 파트 표시
            st.header("OK Parts")
            for part in result_data["ok_parts"]:
                part_number = part[0]["part_number"]
                st.subheader(f"OK Part {part_number}")
                cols = st.columns(5)  # 5개씩 출력
                for idx, img_info in enumerate(part):
                    image_url = img_info["image_url"]
                    image_key = image_url.replace(f"s3://{bucket_name}/", "")
                    img = fetch_image_from_s3(bucket_name, image_key)
                    cols[idx % 5].image(img, caption=f"Part {part_number} - Image {idx + 1}")

        except Exception as e:
            st.error(f"결과를 가져오는 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    view_results_from_s3()

# 언어 초기화 및 선택
init_language()
set_language()
current_language = st.session_state["language"]
text = translations[current_language]["history"]

# history 페이지 내용
st.title(text["title"])
st.subheader(text["description"])
st.write(text["select_history"])