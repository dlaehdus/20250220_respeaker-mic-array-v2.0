import time
import google.generativeai as genai
from threading import Thread, Event
from respeaker import Microphone
import speech_recognition as sr

GEMINI_API_KEY = "AIzaSyB2gskEsPR4ibqbteiaYb9QUs7XhMTBz9U"
genai.configure(api_key=GEMINI_API_KEY)

def select_best_model():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(f"[시스템] 사용 가능 모델 목록: {models}")
        
        for preferred in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']:
            if preferred in models:
                print(f"[시스템] 선택된 모델: {preferred}")
                return genai.GenerativeModel(preferred)
        
        if models:
            return genai.GenerativeModel(models[0])
    except Exception as e:
        print(f"[시스템] 모델 목록 확인 중 오류: {e}")
    return None

model = select_best_model()
if model:
    chat = model.start_chat(history=[])
else:
    print("[시스템] 사용 가능한 Gemini 모델을 찾지 못했습니다.")

def get_gemini_response(user_text):
    try:
        response = chat.send_message(user_text)
        return response.text
    except Exception as e:
        return f"Gemini 대화 오류: {e}"

def task(quit_event):
    mic = Microphone(quit_event=quit_event)
    recognizer = sr.Recognizer()
    
    print("\n" + "="*50)
    print("[시스템] 준비 완료! 'respeaker'라고 부르고 말씀하세요.")
    print("="*50)

    while not quit_event.is_set():
        if mic.wakeup('respeaker'):
            print("\n[나] (듣는 중...)")
            
            # 음성 캡처 및 병합
            audio_generator = mic.listen()
            full_data = b"".join(list(audio_generator)) 

            try:
                # ReSpeaker v2.0 사양
                audio_data = sr.AudioData(full_data, 16000, 2)
                user_input = recognizer.recognize_google(audio_data, language='ko-KR')
                
                if len(user_input.strip()) > 1:
                    print(f"[나] {user_input}")
                    print("[Gemini] 생각 중...")
                    ai_response = get_gemini_response(user_input)
                    print(f"[Gemini] {ai_response}")
                    print("\n" + "-"*30)
                
            except sr.UnknownValueError:
                print("[시스템] 목소리 인식 실패")
            except Exception as e:
                print(f"[시스템] 처리 오류: {e}")

def main():
    if not model: return
    quit_event = Event()
    thread = Thread(target=task, args=(quit_event,))
    thread.daemon = True
    thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n[시스템] 종료합니다.')
        quit_event.set()

if __name__ == '__main__':
    main()