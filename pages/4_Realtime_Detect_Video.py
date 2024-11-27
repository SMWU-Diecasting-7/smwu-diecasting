import os
import cv2
import streamlit as st
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


# 영상 이미지 처리 함수
def realtime_process_video(video_path, tolerance=5, frame_interval=2):
    cap = cv2.VideoCapture(video_path)
    prev_hash = None

    frame_index = 0  # 프레임 번호
    part_number = 1  # 부품 번호
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_part_images = []  # 현재 부품의 이미지 저장
    ng_detect, ok_detect = {}, {}

    # 실시간 이미지 출력용 컨테이너
    realtime_container = st.empty()
    # 진행 바
    progress_bar = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝에 도달한 경우

        # 프레임 샘플링
        if frame_index % frame_interval == 0:
            current_hash = get_image_hash(frame)

            # 중복 프레임 확인
            if prev_hash is None or (
                tolerance < hamming_distance(prev_hash, current_hash) < 40
            ):
                prev_hash = current_hash

                # OpenCV 이미지 전처리
                processed_img = resize_and_pad_image(
                    crop_image(
                        apply_color_jitter(frame, brightness=1.0, contrast=1.0), 1.0
                    )
                )

                # SageMaker 추론
                result = invoke_sagemaker_endpoint(
                    "diecasting-model-T7-endpoint", processed_img
                )
                label = "OK" if result == 1 else "NG"
                label_color = (0, 255, 0) if result == 1 else (0, 0, 255)
                bordered_frame = add_border(frame, label_color)

                # 현재 부품 이미지 추가
                current_part_images.append((bordered_frame, label))

                # 실시간으로 이미지 출력
                with realtime_container.container():
                    st.markdown(f"### No. {part_number}")
                    cols = st.columns(5)
                    for idx, (img, lbl) in enumerate(current_part_images):
                        cols[idx].image(
                            img,
                            channels="BGR",
                            caption=f"Channel {idx + 1}: {lbl}",
                        )

                # 부품 상태 확인 (5개의 이미지가 모두 채워지면)
                if len(current_part_images) == 5:
                    part_status = (
                        "OK"
                        if "NG" not in [lbl for _, lbl in current_part_images]
                        else "NG"
                    )

                    # 부품 상태 출력
                    with realtime_container.container():
                        st.markdown(f"### No. {part_number} - {part_status}")

                    # 최종 이미지 저장
                    if part_status == "NG":
                        ng_detect[part_number] = [
                            img for img, lbl in current_part_images
                        ]
                    else:
                        ok_detect[part_number] = [
                            img for img, lbl in current_part_images
                        ]

                    # 다음 부품을 위해 초기화
                    current_part_images = []
                    part_number += 1

        # 진행 바 업데이트
        progress_bar.progress(frame_index / total_frames)
        frame_index += 1

    cap.release()
    progress_bar.empty()
    realtime_container.empty()
    return ng_detect, ok_detect


# 결과 표시 함수
def show_result_details(detect, status):
    container = st.container()
    with container:
        st.subheader(f"{status} Detailed Images")
        selected_part = st.selectbox(
            f"Select Part to View {status} Images", options=list(detect.keys())
        )
    if selected_part:
        st.write(f"Showing {status} Images for Part {selected_part}")
        cols = st.columns(5)
        for idx, image in enumerate(detect[selected_part]):
            cols[idx % 5].image(
                image,
                channels="BGR",
                caption=f"Part {selected_part} - Channel {idx + 1}",
            )


# 메인 함수
def realtime_video_inference():
    st.title("Real-time NG/OK Detection with Video")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    # 세션 상태 초기화
    if "ng_detect" not in st.session_state:
        st.session_state["ng_detect"] = {}
    if "ok_detect" not in st.session_state:
        st.session_state["ok_detect"] = {}

    if uploaded_file is not None:
        temp_video_path = f"temp_{uploaded_file.name}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Complete Upload File : {uploaded_file.name}")
        st.video(uploaded_file, autoplay=True, muted=True)

        # 추론 결과가 없는 경우에만 추론 실행
        if not st.session_state["ng_detect"] and not st.session_state["ok_detect"]:
            (
                st.session_state["ng_detect"],
                st.session_state["ok_detect"],
            ) = realtime_process_video(temp_video_path, tolerance=5)

        # 결과 출력
        st.subheader("Final Result Summary")
        st.error(f"Total NG Parts: {list(st.session_state['ng_detect'].keys())}")
        st.success(f"Total OK Parts: {list(st.session_state['ok_detect'].keys())}")

        # 상세 결과 표시
        if len(st.session_state["ng_detect"]) > 0:
            show_result_details(st.session_state["ng_detect"], "NG")
        if len(st.session_state["ok_detect"]) > 0:
            show_result_details(st.session_state["ok_detect"], "OK")

        # 삭제: 분석이 끝난 후 임시 파일 삭제
        if os.path.exists(temp_video_path):  # 파일이 존재하는지 확인
            os.remove(temp_video_path)
            # st.info("Temporary video file deleted.")


if __name__ == "__main__":
    realtime_video_inference()
