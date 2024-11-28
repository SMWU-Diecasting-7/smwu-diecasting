import os
import cv2
import streamlit as st
import asyncio
import boto3
import time
from concurrent.futures import ThreadPoolExecutor
from translations import init_language, set_language, translations
from utils import (
    get_image_hash,
    hamming_distance,
    resize_and_pad_image,
    crop_image,
    apply_color_jitter,
    add_border,
    invoke_sagemaker_endpoint,
)

st.set_page_config(page_title="Realtime Detect Video", page_icon="📹")

# 언어 초기화 및 선택
init_language()
set_language()
current_language = st.session_state["language"]
text = translations[current_language]["video"]


# 비동기로 SageMaker 호출
async def async_invoke_sagemaker(frame_idx, processed_img, loop, executor):
    result = await loop.run_in_executor(
        executor,
        invoke_sagemaker_endpoint,
        "diecasting-model-T7-endpoint",
        processed_img,
    )
    return frame_idx, result


async def realtime_process_video_async(video_path, tolerance=5, frame_interval=2):
    cap = cv2.VideoCapture(video_path)
    prev_hash = None

    frame_index = 0
    part_number = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_part_images = []  # 현재 부품 이미지 저장
    ng_detect = {}
    ok_detect = {}

    # ThreadPoolExecutor 생성
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()

    # 병렬 작업 관리용 Semaphore
    semaphore = asyncio.Semaphore(10)  # 최대 동시 Task 수 제한

    # 실시간 이미지 출력용 컨테이너
    realtime_container = st.empty()

    async def process_frame(frame, frame_idx):
        """단일 프레임 처리"""
        nonlocal prev_hash, current_part_images, part_number

        current_hash = get_image_hash(frame)

        # 중복 프레임 제거
        if prev_hash is None or (
            tolerance < hamming_distance(prev_hash, current_hash) < 40
        ):
            prev_hash = current_hash

            # opencv 이미지 전처리
            processed_img = resize_and_pad_image(
                crop_image(apply_color_jitter(frame, brightness=1.0, contrast=1.0), 1.0)
            )

            # 비동기로 SageMaker 호출
            frame_idx, result = await async_invoke_sagemaker(
                frame_idx, processed_img, loop, executor
            )

            label = "OK" if result == 1 else "NG"
            label_color = (0, 255, 0) if result == 1 else (0, 0, 255)
            bordered_frame = add_border(frame, label_color)

            # 응답 저장
            current_part_images.append((frame_idx, bordered_frame, label))
            # 프레임 번호 기준으로 정렬
            current_part_images.sort(key=lambda x: x[0])

            # 실시간으로 이미지 출력
            with realtime_container.container():
                st.markdown(f"### No. {part_number}")
                cols = st.columns(5)
                for idx, (_, img, lbl) in enumerate(current_part_images):
                    cols[idx].image(
                        img,
                        channels="BGR",
                        caption=f"Channel {idx + 1}: {lbl}",
                    )

            # 부품 상태 확인 (5개의 이미지가 모두 채워지면)
            if len(current_part_images) == 5:
                part_status = (
                    "OK"
                    if "NG" not in [lbl for _, _, lbl in current_part_images]
                    else "NG"
                )

                # 최종 이미지 저장
                if part_status == "NG":
                    ng_detect[part_number] = [
                        (img, lbl) for _, img, lbl in current_part_images
                    ]
                else:
                    ok_detect[part_number] = [
                        (img, lbl) for _, img, lbl in current_part_images
                    ]

                # 부품 상태 출력
                with realtime_container.container():
                    st.markdown(f"### No. {part_number} - {part_status}")
                    cols = st.columns(5)
                    for idx, (_, img, lbl) in enumerate(current_part_images):
                        cols[idx].image(
                            img,
                            channels="BGR",
                            caption=f"Channel {idx + 1}: {lbl}",
                        )

                # 초기화
                current_part_images = []
                realtime_container.empty()
                part_number += 1

    # 제한된 프레임 처리 함수 (병목 방지)
    async def limited_process_frame(frame, frame_idx):
        async with semaphore:
            return await process_frame(frame, frame_idx)

    # 실행 함수
    tasks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝에 도달한 경우

        # 프레임 샘플링
        if frame_index % frame_interval == 0:
            tasks.append(limited_process_frame(frame, frame_index))

        frame_index += 1

    await asyncio.gather(*tasks)

    cap.release()
    realtime_container.empty()
    return ng_detect, ok_detect


@st.cache_data
def get_cached_images(detect, part_number):
    return detect[part_number]


def show_result_details(detect, status):
    container = st.container()
    with container:
        st.subheader(f"{status} {text["detailed_image"]}")
        selected_part = st.selectbox(
            f"{text["select_img_box"]} : {status}",
            options=list(detect.keys()),
            key=f"select_{status}",
        )

    if selected_part:
        st.subheader(f"No. {selected_part}")
        images = get_cached_images(detect, selected_part)
        cols = st.columns(5)
        for idx, (image, label) in enumerate(images):
            cols[idx % 5].image(
                image,
                channels="BGR",
                caption=f"Part {selected_part} - Channel {idx + 1}: {label}",
            )


def upload_results_to_s3(ng_detect, ok_detect):
    s3 = boto3.client("s3")

    upload_time = time.strftime("%Y%m%d_%H%M%S")


# 메인 함수
def realtime_video_inference():
    st.title(text["title"])

    uploaded_file = st.file_uploader(text["upload"], type=["mp4", "mov", "avi"])

    # 세션 상태 초기화
    if "upload_time" not in st.session_state:
        st.session_state.upload_time = None  # 업로드 시간 저장
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False  # 분석 상태 추적

    if uploaded_file is not None:
        current_upload_time = time.strftime("%Y%m%d_%H%M%S")  # 현재 업로드 시간
        # 새 파일 업로드 이벤트 처리
        if st.session_state.upload_time != current_upload_time:
            # 상태 초기화
            st.session_state.upload_time = current_upload_time
            st.session_state.analysis_done = False

        temp_video_path = f"temp_{uploaded_file.name}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(
            f"{text["upload_success"]} : {uploaded_file.name} ({st.session_state['upload_time']})"
        )
        st.video(temp_video_path, autoplay=True, muted=True)

        if not st.session_state.analysis_done:
            with st.spinner(text["processing"]):
                (
                    ng_detect,
                    ok_detect,
                ) = asyncio.run(
                    realtime_process_video_async(temp_video_path, tolerance=5)
                )
                st.session_state.analysis_done = True

        # 결과 출력
        st.subheader(text["summary"])
        if len(ng_detect.keys()) > 0:
            if current_language == "en":
                st.error(
                    f"{text['total']} {len(ng_detect.keys())} NG {text['parts']}: {list(ng_detect.keys())}"
                )
            elif current_language == "kr":
                st.error(
                    f"{text['total']} {len(ng_detect.keys())}개의 NG {text['parts']}: {list(ng_detect.keys())}"
                )
        if len(ok_detect.keys()) > 0:
            if current_language == "en":
                st.success(
                    f"{text['total']} {len(ok_detect.keys())} OK {text['parts']}: {list(ok_detect.keys())}"
                )
            elif current_language == "kr":
                st.success(
                    f"{text['total']} {len(ok_detect.keys())}개의 OK {text['parts']}: {list(ok_detect.keys())}"
                )

        @st.fragment
        def show_ng_section():
            if len(ng_detect) > 0:
                show_result_details(ng_detect, "NG")

        @st.fragment
        def show_ok_section():
            if len(ok_detect) > 0:
                show_result_details(ok_detect, "OK")

        show_ng_section()
        show_ok_section()

        # 삭제: 분석이 끝난 후 임시 파일 삭제
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


if __name__ == "__main__":
    realtime_video_inference()
