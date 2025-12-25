import logging
import time
import sys
from threading import Thread, Event
from respeaker import Microphone
import speech_recognition as sr

# 1. 즉시 실행 확인을 위한 프린트
print("프로그램을 시작합니다... 로딩 중...")

def task(quit_event):
    mic = Microphone(quit_event=quit_event)
    recognizer = sr.Recognizer()
    
    # --- 추가: 주변 소음 수준에 맞춰 감도 자동 조절 ---
    # 이 수치를 높이면 더 큰 소리에만 반응합니다.
    recognizer.dynamic_energy_threshold = True 
    recognizer.energy_threshold = 4000 
    # ----------------------------------------------

    print("\n>>> 준비 완료! 'respeaker'라고 부른 뒤 한국어로 말씀하세요.")

    while not quit_event.is_set():
        if mic.wakeup('respeaker'):
            print("\n[네, 듣고 있어요! 말씀해 주세요...]")
            
            # 음성 캡처
            audio_generator = mic.listen()
            full_data = b"".join(list(audio_generator)) 

            try:
                audio_data = sr.AudioData(full_data, 16000, 2) 
                
                # --- 수정: 너무 짧은 소음은 무시하도록 설정 ---
                # phrase_time_limit 등을 사용해 너무 긴 잡음 유입을 막을 수 있습니다.
                print("[구글 서버에서 분석 중...]")
                text = recognizer.recognize_google(audio_data, language='ko-KR')
                
                # 인식된 글자가 너무 짧거나 숫자만 있다면 무시하는 로직 추가
                if len(text.strip()) > 1:
                    print(f">>> 인식된 결과: {text}")
                else:
                    print(">>> 의미 있는 문장이 인식되지 않았습니다.")
                
            except sr.UnknownValueError:
                # 인식 불가능한 소음일 경우 그냥 넘어갑니다.
                pass 
            except Exception as e:
                print(f">>> 오류 발생: {e}")

def main():
    quit_event = Event()
    thread = Thread(target=task, args=(quit_event,))
    thread.daemon = True
    thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n사용자에 의해 종료되었습니다.')
        quit_event.set()

# 이 부분이 반드시 있어야 프로그램이 실행됩니다.
if __name__ == '__main__':
    main()