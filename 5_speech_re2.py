import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import requests
from threading import Thread, Event, Lock
from respeaker import Microphone
import speech_recognition as sr

# ===============================
# 1. 시각 인식 설정 (RealSense & YOLO)
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
    print("[시스템] 카메라 및 센서 준비 완료")
except Exception as e:
    print(f"[오류] 카메라 초기화 실패: {e}")

yolo_model = YOLO("yolov8n-seg.pt") 
yolo_model.fuse()

current_seen_objects = []
data_lock = Lock()
ACTION_LIST = ["blue button", "cup", "mouse pick", "open door", "pick and place"]

# ===============================
# 2. 로컬 AI 설정 (Ollama)
# ===============================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2" 

def ask_fast_ai(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "1h",
        "options": {
            "num_predict": 150,
            "temperature": 0.3
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=10)
        return response.json().get("response", "")
    except Exception as e:
        return f"로컬 AI 통신 에러: {e}"

# ===============================
# 3. 실시간 카메라 스레드
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
            results = yolo_model.track(frame, conf=0.4, persist=True, verbose=False)

            temp_objects = []
            if results[0].boxes is not None:
                for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
                    label = yolo_model.names[int(results[0].boxes.cls[i])]
                    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    depth = depth_frame.get_distance(cx, cy)
                    
                    if 0.01 < depth < 3.0: 
                        p = rs.rs2_deproject_pixel_to_point(color_intrinsics, [cx, cy], depth)
                        side = "오른쪽" if p[0] > 0.05 else ("왼쪽" if p[0] < -0.05 else "정면")
                        temp_objects.append(f"{label}(위치:{side}, 거리:{p[2]:.1f}m)")

            with data_lock:
                current_seen_objects = temp_objects

            cv2.imshow("Robot Vision (Local LLM)", results[0].plot())
            if cv2.waitKey(1) & 0xFF == 27: break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# ===============================
# 4. 음성 인식 및 AI 판단 스레드
# ===============================
def voice_task(quit_event):
    mic = Microphone(quit_event=quit_event)
    recognizer = sr.Recognizer()

    print(f"\n[알림] 로컬 AI '{MODEL_NAME}'가 대기 중입니다.")
    print("[알림] 'respeaker'이라고 부르고 명령하세요.") 

    while not quit_event.is_set():
        # 1. 호출어 인식 대기
        if mic.wakeup('respeaker'): 
            # 호출어를 인식했을 때 즉시 출력
            print("[respeaker]: 대답하세요.")
            
            # 2. 사용자 명령 듣기 시작
            audio_gen = mic.listen()
            full_data = b"".join(list(audio_gen))

            try:
                audio_data = sr.AudioData(full_data, 16000, 2)
                user_input = recognizer.recognize_google(audio_data, language='ko-KR')
                print(f"[사용자]: {user_input}")

                with data_lock:
                    vision_context = ", ".join(current_seen_objects) if current_seen_objects else "인식된 물체 없음"

                prompt = f"""
현재 로봇의 시각 정보: {vision_context}
가능한 액션 리스트: {ACTION_LIST}

사용자 질문: "{user_input}"

위 정보를 바탕으로 한국어로 답변해줘. 
관련된 액션이 있다면 답변 끝에 'SELECTED_ACTION: [액션명]' 형식을 반드시 포함해."""

                print("\n" + "="*20 + " [AI 전송 데이터 상세] " + "="*20)
                print(prompt)
                print("="*60)

                start_time = time.time()
                response = ask_fast_ai(prompt)
                end_time = time.time()
                
                print(f"\n[가넷 답변 (소요시간: {end_time-start_time:.2f}초)]")
                print(f"{response}")
                print("-" * 50)

            except sr.UnknownValueError:
                print("[시스템] 음성을 이해하지 못했습니다. 다시 말씀해주세요.")
            except Exception as e:
                print(f"[오류] 데이터 처리 중 에러 발생: {e}")

# ===============================
# 5. 메인 실행 루프
# ===============================
if __name__ == '__main__':
    quit_event = Event()
    t_cam = Thread(target=camera_thread_func, args=(quit_event,))
    t_voice = Thread(target=voice_task, args=(quit_event,))
    
    t_cam.start()
    t_voice.start()

    try:
        while not quit_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[시스템] 종료 요청을 받았습니다.")
        quit_event.set()
    finally:
        t_cam.join()
        t_voice.join()
        print("[시스템] 프로그램이 안전하게 종료되었습니다.")