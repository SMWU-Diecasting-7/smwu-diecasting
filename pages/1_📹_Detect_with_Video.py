import cv2
import streamlit as st
from utils import get_image_hash, hamming_distance, resize_and_pad_image, crop_image, apply_color_jitter, add_border, invoke_sagemaker_endpoint
import torchvision.transforms as transforms
import numpy as np

# 이미지 전처리 - transforms
preprocess = transforms.Compose([
    transforms.ToPILImage(),  # OpenCV 이미지(Numpy 배열)를 PIL 이미지로 변환
    transforms.ToTensor(),    # 텐서로 변환
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
])

# 페이지 설정
st.set_page_config(
    page_title="Detect with Video",
    page_icon="📹",
)

# 영상 이미지 처리 함수
def process_video(video_path, tolerance=5):
    cap = cv2.VideoCapture(video_path)
    prev_hash = None
    unique_images = []
    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝에 도달한 경우
        
        # 현재 프레임의 해시 값 계산
        current_hash = get_image_hash(frame)
        
        if prev_hash is None or (tolerance < hamming_distance(prev_hash, current_hash) < 40):
            # opencv 이미지 전처리
            processed_img = resize_and_pad_image(
                crop_image(apply_color_jitter(frame, brightness=1.3, contrast=1.5), 1.0)
            )
            # 2. torch 이미지 전처리 (PIL 변환 -> 텐서 변환 -> 정규화)
            processed_img_tensor = preprocess(processed_img)  # 텐서화 및 정규화
            processed_img_numpy = (processed_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # HWC 변환
            unique_images.append(processed_img_numpy)  # NumPy 배열로 저장
        
        prev_hash = current_hash
        frame_index += 1
        progress_bar.progress(frame_index / total_frames)
    
    cap.release()
    progress_bar.empty()
    return unique_images
    
# 결과 표시 함수 (5개씩 묶어서 표시)
def display_results(unique_images, results):
    st.subheader("Predict Result")
    
    ng_images = []
    ng_No = set()
    ok_No = set()

    # 5개씩 묶어서 처리
    for i in range(0, len(results), 5):
        batch_results = results[i:i+5]  # 현재 묶음 결과
        batch_images = unique_images[i:i+5]  # 현재 묶음 이미지
        
        # 부품 상태 결정 (하나라도 NG이면 전체 NG)
        batch_status = "NG" if 0 in batch_results else "OK"
        color = (0, 255, 0) if batch_status == "OK" else (0, 0, 255)
        
        # 최종 결과 표시용
        part_number = i // 5 + 1
        if batch_status == "NG":
            ng_No.add(part_number)
        else:
            ok_No.add(part_number)

        # 부품 상태 표시
        st.markdown(f"### No. {i//5 + 1}: **{batch_status}**")

        cols = st.columns(5)
        # 각 이미지에 결과 표시
        for j, (image, status) in enumerate(zip(batch_images, batch_results)):
            label = "OK" if status == 1 else "NG"
            label_color = (0, 255, 0) if status == 1 else (0, 0, 255)
            
            # NG 이미지 저장
            if status == 0:
                ng_images.append((image, i + j))
            
            bordered_image = add_border(image, label_color)
            part_number = (i + j) // 5 + 1
            channel_number = (i + j) % 5 + 1
            cols[j].image(
                bordered_image,
                channels="BGR",
                caption=f"Part {part_number} - Channel {channel_number} ({label})",
            )
    # NG 이미지만 추가 출력
    if ng_images:
        st.subheader("Final NG Images")
        cols = st.columns(5)
        for idx, (ng_image, ng_index) in enumerate(ng_images):
            part_number = ng_index // 5 + 1
            channel_number = ng_index % 5 + 1
            
            bordered_ng_image = add_border(ng_image, (0, 0, 255))
            
            # 이미지 출력: 부품 번호와 채널 번호 표시
            cols[idx % 5].image(
                bordered_ng_image, 
                channels="BGR", 
                caption=f"No. {part_number} - Channel {channel_number}"
            )
    
    # 최종 NG/OK 부품 번호 출력
    st.subheader("Final Result Summary")
    if ng_No:
        st.error(f"NG Parts: {', '.join(map(str, ng_No))} (Total: {len(ng_No)})")
    if ok_No:
        st.success(f"OK Parts: {', '.join(map(str, ok_No))} (Total: {len(ok_No)})")
            

# 메인 함수
def video_inference():
    # Streamlit 애플리케이션
    st.title("Real-time NG/OK Video Classification")
    
    # 고유 프레임 저장용 세션 상태 초기화
    if "unique_images" not in st.session_state:
        st.session_state["unique_images"] = []
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # 업로드된 비디오를 임시 파일로 저장
        temp_video_path = f"temp_{uploaded_file.name}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Complete Upload File : {uploaded_file.name}")
    
        # 업로드 출력
        st.subheader("Uploaded Video")
        st.video(temp_video_path)
    
        # 영상 이미지 추출
        with st.spinner("Extracting images from video..."):
            unique_images = process_video(temp_video_path, tolerance=5)
            st.session_state["unique_images"] = unique_images  # 세션 상태에 저장
            st.success(f"Total {len(unique_images)} images extraction")
        
        # SageMaker 분석
        with st.spinner("Analyzing images with SageMaker..."):
            progress_bar = st.progress(0)  # 진행바
            status_text = st.empty()       # 상태 메시지 표시용
            
            results = []
            for i, image in enumerate(st.session_state["unique_images"]):
                status_text.text(f"Processing image {i + 1}/{len(st.session_state["unique_images"])}")
                
                result = invoke_sagemaker_endpoint('test-endpoint', image)
                results.append(result)    
                
                # 진행률 업데이트
                progress_bar.progress((i + 1) / len(st.session_state["unique_images"]))  
                      
            progress_bar.empty()
            status_text.text("All images processed!")
            st.session_state["results"] = results
        
        # 분석 결과 표시
        display_results(st.session_state["unique_images"], results)
                
# 프로그램 실행
if __name__ == "__main__":
    video_inference()
