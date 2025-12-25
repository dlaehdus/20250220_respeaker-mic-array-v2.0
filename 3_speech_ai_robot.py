import time
import google.generativeai as genai
from threading import Thread, Event
from respeaker import Microphone
import speech_recognition as sr

# --- 1. Gemini 설정 및 API 키 ---
GEMINI_API_KEY = "AIzaSyB2gskEsPR4ibqbteiaYb9QUs7XhMTBz9U"
genai.configure(api_key=GEMINI_API_KEY)

# 매핑할 후보 단어 리스트
ACTION_LIST = ["blue button", "red button", "green button", "open door", "pick and place"]

def select_best_model():
    """사용 가능한 최적의 모델을 자동으로 선택"""
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(f"[시스템] 사용 가능 모델 목록: {models}")
        
        for preferred in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']:
            if preferred in models:
                print(f"[시스템] 선택된 모델: {preferred}")
                return genai.GenerativeModel(preferred)
        
        if models:
            print(f"[시스템] 기본 모델 선택: {models[0]}")
            return genai.GenerativeModel(models[0])
    except Exception as e:
        print(f"[시스템] 모델 목록 확인 중 오류: {e}")
    return None

# 모델 초기화
model = select_best_model()

def get_best_action(user_text):
    """Gemini를 사용하여 사용자의 말과 가장 연관된 단어를 선택하여 변수로 반환"""
    if not model:
        return "error"

    prompt = f"""
    당신은 명령어 분석 시스템입니다. 사용자의 입력을 분석하여 아래 후보 리스트 중 가장 연관성이 높은 단어 하나만 골라주세요.
    리스트에 없는 단어는 절대 말하지 말고, 오직 후보 리스트의 단어 중 하나만 출력하세요.
    연관된 것이 없다면 'none'이라고 답하세요.

    후보 리스트: {', '.join(ACTION_LIST)}

    사용자 입력: "{user_text}"
    결과:"""
    
    try:
        response = model.generate_content(prompt)
        selected_action = response.text.strip().lower()
        
        # 실제 리스트에 존재하는 단어인지 최종 검증
        for action in ACTION_LIST:
            if action in selected_action:
                return action
        return "none"
    except Exception as e:
        print(f"[시스템] Gemini 호출 오류: {e}")
        return "error"

def task(quit_event):
    mic = Microphone(quit_event=quit_event)
    recognizer = sr.Recognizer()
    
    print("\n" + "="*50)
    print("[시스템] 준비 완료! 'respeaker'라고 부르고 말씀하세요.")
    print(f"[감지 대상]: {ACTION_LIST}")
    print("="*50)

    while not quit_event.is_set():
        if mic.wakeup('respeaker'):
            print("\n[나] (듣는 중...)")
            
            audio_generator = mic.listen()
            full_data = b"".join(list(audio_generator)) 

            try:
                audio_data = sr.AudioData(full_data, 16000, 2)
                user_input = recognizer.recognize_google(audio_data, language='ko-KR')
                
                if len(user_input.strip()) > 1:
                    print(f"[나] {user_input}")
                    print("[분석 중...]")
                    
                    # 1. 연관된 단어를 찾아서 변수에 저장
                    final_action = get_best_action(user_input)
                    
                    # 2. 결과 출력
                    print(f"\n>>> [저장된 변수 값]: {final_action}")
                    
                    # 3. 변수 값에 따른 후속 동작 예시
                    if final_action == "blue button":
                        print("동작: 파란색 버튼 관련 프로세스를 실행합니다.")
                    elif final_action == "open door":
                        print("동작: 문을 여는 하드웨어 신호를 보냅니다.")
                    elif final_action == "pick and place":
                        print("동작: 물건을 집어서 옮기는 로봇 팔 동작을 시작합니다.")
                    
                    print("\n" + "-"*30)
                
            except sr.UnknownValueError:
                print("[시스템] 목소리 인식 실패")
            except Exception as e:
                print(f"[시스템] 처리 오류: {e}")

def main():
    if not model:
        print("[시스템] 모델 로드에 실패하여 프로그램을 종료합니다.")
        return
    
    quit_event = Event()
    thread = Thread(target=task, args=(quit_event,))
    thread.daemon = True
    thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n[시스템] 프로그램을 종료합니다.')
        quit_event.set()

if __name__ == '__main__':
    main()