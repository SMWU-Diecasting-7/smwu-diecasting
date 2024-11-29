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
                try:
                    json_data = s3.get_object(Bucket=bucket_name, Key=key)["Body"].read()
                    results.append(json.loads(json_data))
                except json.JSONDecodeError:
                    st.error(f"Failed to decode JSON for {key}")
                except Exception as e:
                    st.error(f"Error fetching data for {key}: {str(e)}")
    return results

# 히스토리 데이터를 화면에 표시
def display_history(results):
    if not results:
        st.write("No history found.")
        return

    for result in results:
        if isinstance(result, dict) and "video_name" in result and "ng_parts" in result and "ok_parts" in result:
            with st.expander(f"Results for {result['video_name']}"):
                st.subheader(f"Video Name: {result['video_name']}")

                # NG 이미지 출력
                st.error("NG Parts:")
                if isinstance(result["ng_parts"], list) and result["ng_parts"]:
                    for part in result["ng_parts"]:
                        if isinstance(part, dict) and "image_url" in part:
                            st.markdown(
                                f"<div style='background-color:#FFCCCC;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                f"<b>NG Part {part.get('part_number', 'Unknown')}:</b><br>"
                                f"<img src='{part['image_url']}' style='max-width:100%;border-radius:5px;' />"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # part가 문자열로 처리될 수 있도록 변환
                            st.markdown(
                                f"<div style='background-color:#FFCCCC;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                f"Invalid NG Part Data: {str(part)}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.write("No NG Parts.")

                # OK 이미지 출력
                st.success("OK Parts:")
                if isinstance(result["ok_parts"], list) and result["ok_parts"]:
                    for part in result["ok_parts"]:
                        if isinstance(part, dict) and "image_url" in part:
                            st.markdown(
                                f"<div style='background-color:#CCFFCC;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                f"<b>OK Part {part.get('part_number', 'Unknown')}:</b><br>"
                                f"<img src='{part['image_url']}' style='max-width:100%;border-radius:5px;' />"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            # part가 문자열로 처리될 수 있도록 변환
                            st.markdown(
                                f"<div style='background-color:#CCFFCC;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                f"Invalid OK Part Data: {str(part)}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.write("No OK Parts.")
        else:
            st.error("Invalid data format detected in history results.")


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
        st.success("History loaded")
        display_history(results)

if __name__ == "__main__":
    main()
