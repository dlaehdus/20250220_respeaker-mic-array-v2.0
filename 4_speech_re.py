import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import google.generativeai as genai
from threading import Thread, Event, Lock
from respeaker import Microphone
import speech_recognition as sr

# ===============================
# 1. 초기 설정 (RealSense & YOLO)
# ===============================
try:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    align = rs.align(rs.stream.color)
    profile = pipeline.get_active_profile()
    color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
except Exception as e:
    print(f"[오류] RealSense 초기화 실패: {e}")

yolo_model = YOLO("yolov8m-seg.pt")
yolo_model.fuse()

current_seen_objects = []
data_lock = Lock()

# ===============================
# 2. Gemini 설정 (안정화 버전)
# ===============================
GEMINI_API_KEY = "AIzaSyBzwVeJW3ShE-S49ULxp2MAx0DCOCKs800"  # 발급받은 키를 여기에 입력하세요
genai.configure(api_key=GEMINI_API_KEY)

ACTION_LIST = ["blue button", "cup", "mouse pick", "open door", "pick and place"]

def select_best_model():
    try:
        # 사용 가능한 모델 목록을 가져옵니다.
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(f"[시스템] 사용 가능 모델 목록: {available_models}")
        
        # 가장 안정적인 모델 순서로 선택
        target_models = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        for target in target_models:
            if target in available_models:
                print(f"[시스템] 최종 선택된 모델: {target}")
                return genai.GenerativeModel(target)
        
        if available_models:
            return genai.GenerativeModel(available_models[0])
    except Exception as e:
        print(f"[오류] 모델 선택 중 문제 발생: {e}")
    return None

gemini_model = select_best_model()

# 중요: 모델이 정상적으로 로드되었는지 확인 후 대화 세션 시작
if gemini_model:
    chat = gemini_model.start_chat(history=[])
else:
    print("[치명적 오류] Gemini 모델을 불러오지 못했습니다. API 키나 인터넷 연결을 확인하세요.")
    exit() # 프로그램 종료

# ===============================
# 3. 카메라 스레드 (실시간 인식)
# ===============================
def camera_thread_func(quit_event):
    global current_seen_objects
    try:
        while not quit_event.is_set():
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not color_frame or not depth_frame: continue

            frame = np.asanyarray(color_frame.get_data())
            results = yolo_model.track(frame, conf=0.25, persist=True, verbose=False)

            temp_objects = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    label = yolo_model.names[int(classes[i])]
                    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    
                    depth = depth_frame.get_distance(cx, cy)
                    if depth > 0:
                        p = rs.rs2_deproject_pixel_to_point(color_intrinsics, [cx, cy], depth)
                        side = "오른쪽" if p[0] > 0.05 else ("왼쪽" if p[0] < -0.05 else "정면")
                        temp_objects.append(f"{label}: {side} {p[2]:.2f}m (x:{p[0]*100:.1f}cm, y:{p[1]*100:.1f}cm)")

            with data_lock:
                current_seen_objects = temp_objects

            cv2.imshow("AI Vision System", results[0].plot())
            if cv2.waitKey(1) & 0xFF == 27: break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# ===============================
# 4. 음성 인식 및 데이터 누적 스레드
# ===============================
def voice_task(quit_event):
    mic = Microphone(quit_event=quit_event)
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    print("\n" + "="*50)
    print("[시스템] 준비 완료! 'respeaker'라고 부르고 말씀하세요.")
    print("="*50)

    while not quit_event.is_set():
        if mic.wakeup('respeaker'):
            print("\n[나] 듣는 중...")
            audio_gen = mic.listen()
            full_data = b"".join(list(audio_gen))

            try:
                audio_data = sr.AudioData(full_data, 16000, 2)
                user_input = recognizer.recognize_google(audio_data, language='ko-KR')
                print(f"[나] {user_input}")

                # 1초간 시각 데이터 누적 수집
                print("[시스템] 주변 상황 정밀 분석 중 (1초)...")
                accumulated_data = set()
                start_obs_time = time.time()
                
                while time.time() - start_obs_time < 1.0:
                    with data_lock:
                        for item in current_seen_objects:
                            accumulated_data.add(item)
                    time.sleep(0.1)
                
                vision_context = ", ".join(list(accumulated_data)) if accumulated_data else "보이는 물체 없음"

                prompt = f"""
                관찰 정보: [{vision_context}]
                액션 리스트: [{', '.join(ACTION_LIST)}]
                질문: "{user_input}"
                너는 시각 로봇이야. 위 정보를 바탕으로 질문에 답하고, 연관 액션이 있다면 'SELECTED_ACTION: [명령어]'를 답 끝에 붙여줘.
                """
                
                print("[Gemini] 대답 생성 중...")
                response = chat.send_message(prompt)
                print(f"[Gemini] {response.text}")

            except Exception as e:
                print(f"[오류] {e}")

# ===============================
# 5. 메인 실행 루프
# ===============================
if __name__ == '__main__':
    quit_event = Event()
    t_cam = Thread(target=camera_thread_func, args=(quit_event,))
    t_voice = Thread(target=voice_task, args=(quit_event,))
    t_cam.start(); t_voice.start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        quit_event.set(); t_cam.join(); t_voice.join()