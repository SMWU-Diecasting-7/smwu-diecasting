import boto3
import json
import streamlit as st
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv(dotenv_path="AWS.env")

# Streamlit 페이지 설정
st.set_page_config(
    page_title="History",
    page_icon="⌛️",
)

# S3 클라이언트 생성
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

# S3에서 JSON 결과 불러오기
def fetch_results_from_s3(bucket_name, prefix):
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    results = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith(".json"):
                json_data = s3.get_object(Bucket=bucket_name, Key=key)["Body"].read()
                results.append(json.loads(json_data))
    return results

# 히스토리 데이터 표시 함수
def display_history(results):
    if not results:
        st.write("No history found.")
        return

    for result in results:
        with st.expander(f"Results for {result['video_name']}"):
            st.subheader(f"Video Name: {result['video_name']}")
            st.error(f"NG Parts: {result['ng_parts']}")
            st.success(f"OK Parts: {result['ok_parts']}")

# 메인 함수
def main():
    st.title("Detection History")

    # S3 버킷 정보
    bucket_name = "cv-7-video"
    results_prefix = "results/"

    # 히스토리 로드 버튼
    if st.button("Load History from S3"):
        with st.spinner("Loading history..."):
            results = fetch_results_from_s3(bucket_name, results_prefix)
        st.success("History loaded!")
        display_history(results)

if __name__ == "__main__":
    main()

