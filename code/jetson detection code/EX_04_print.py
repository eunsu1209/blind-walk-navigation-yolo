import cv2
import time
import threading
import queue
import ollama
from ultralytics import YOLO


# ==========================================
# 1. 초기 설정 (해상도 및 쿨다운)
# ==========================================
CAMW = 640
CAMH = 480
ALERT_COOLDOWN = 5.0 

# ==========================================
# 2. 멀티쓰레딩 큐 및 플래그 세팅
# ==========================================
audio_queue = queue.Queue()
ollama_queue = queue.Queue()

# Ollama 제어 플래그
ollama_is_busy = False
lock = threading.Lock()

print("✅ 구글 TTS 및 오디오 큐 준비 완료 (온라인 전용)")

def audio_worker():
    """백그라운드에서 구글 TTS를 이용해 자연스러운 음성을 출력하는 쓰레드"""
    import os
    from gtts import gTTS
    import time
    
    while True:
        message = audio_queue.get() 
        try:
            print(f"🔊 [구글 TTS 재생 중] : {message}")
            
            # 1. 구글 서버에서 음성 파일 생성
            tts = gTTS(text=message, lang='ko')
            
            # 2. 임시 파일 저장
            tts.save("temp_speech.mp3")
            
            # 3. 리눅스 플레이어로 출력
            os.system("mpg123 -q temp_speech.mp3")
            
            # 4. 임시 파일 임무 교대 후 삭제
            if os.path.exists("temp_speech.mp3"):
                os.remove("temp_speech.mp3")
                
        except Exception as e:
            print(f"❌ [구글 TTS 에러] 재생 중 오류: {e}")
        finally:
            audio_queue.task_done()

# Ollama Context 유지용 변수
ollama_history = [
    {
        'role': 'system', 
        'content': '당신은 시각장애인을 위한 이동 보조 AI입니다. 입력되는 카메라 사물 정보를 바탕으로, 운전자나 보행자가 주의해야 할 핵심 요약 안내를 한국어 한 문장(15자 이내)으로 매우 짧고 직관적으로 답하세요. 예: "정면에 사람과 자동차가 있으니 주의하세요."'
    }
]

def ollama_worker():
    """Ollama 추론 및 완료 시점 제어를 담당하는 전담 쓰레드"""
    global ollama_history, ollama_is_busy
    while True:
        text_to_analyze = ollama_queue.get()
        print(f"\n[데이터 수집] {text_to_analyze}")
        print("🤖 Ollama 상황 분석 중... ", end='', flush=True)

        ollama_history.append({'role': 'user', 'content': f"현재 감지 상태: {text_to_analyze}"})
        
        try:
            stream = ollama.chat(
                model='gemma3:4b',
                messages=ollama_history,
                stream=True,
            )

            ai_response = ""
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                ai_response += content
            print()  # 생성 완료 후 줄바꿈

            # 대화 기록 관리
            ollama_history.append({'role': 'assistant', 'content': ai_response})
            if len(ollama_history) > 6:
                ollama_history = [ollama_history[0]] + ollama_history[-4:]

            # 기존 밀린 오디오 청소
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # 정제된 멘트 전송
            audio_queue.put(ai_response)

        except Exception as ollama_err:
            print(f"\n❌ [Ollama 에러] {ollama_err}")
            if len(ollama_history) > 1:
                ollama_history.pop()
        finally:
            # [핵심] Ollama가 출력을 마쳤으므로 바쁨 플래그를 해제 (카메라가 새로 인식할 수 있도록 허용)
            with lock:
                ollama_is_busy = False
            ollama_queue.task_done()

# 쓰레드 시작
threading.Thread(target=audio_worker, daemon=True).start()
threading.Thread(target=ollama_worker, daemon=True).start()

# ==========================================
# 3. 클래스 이름 정의 및 상태 변수
# ==========================================
custom_names_1 = {0: '빨간불', 1: '초록불'}
coco_classes_filter = [0, 1, 2, 3, 5, 7, 11, 13, 16]
coco_names = {
    0: '사람', 1: '자전거', 2: '자동차', 3: '오토바이',
    5: '버스', 7: '트럭', 11: '정지 표지판', 13: '벤치', 16: '개'
}
custom_names_2 = {
    3: 'Manhole',
    4: 'Bollard',
    5: 'StationShelter'
}
custom_names_3 = {
    2: '횡단보도',
    3: '내려가는계단',
    4: 'greenlight',
    5: 'redlight',
    6: '올라가는계단'
}
last_spoken_text = ""
last_spoken_time = 0.0

# ==========================================
# 4. AI 모델 로드 및 카메라 설정
# ==========================================
custom_model_1 = YOLO("traffic_light_balanced_v3.engine", task="detect")
coco_model = YOLO("rps_yolo11n_coco.engine", task="detect")
custom_model_2 = YOLO("obstacle_v1.engine", task="detect")
custom_model_3 = YOLO("navigation-1_yolo11n_6402.engine", task="detect")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMH)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cam', CAMW + 40, CAMH + 60)

prev_time = time.time()
fps = 0.0

# ==========================================
# 5. 탐지된 객체 추출 함수
# ==========================================
def get_detections(results, name_dict):
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0].item()) 
        confidence = box.conf[0].item()   
        if class_id in name_dict:
            detections.append({
                'class_name': name_dict[class_id],
                'conf': confidence,
                'box': box.xyxy[0]
            })
    return detections

# ==========================================
# 6. 메인 비디오 루프
# ==========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    current_time = time.time()

    # YOLO 추론
    results_custom = custom_model_1(frame, imgsz=640, conf=0.5, iou=0.45, device=0, verbose=False)
    results_custom[0].names = custom_names_1

    results_coco = coco_model(frame, imgsz=320, conf=0.5, iou=0.45, device=0, verbose=False, classes=coco_classes_filter)
    results_coco[0].names = coco_names

    # 모델 3: 장애물(시설물)
    results_custom_2 = custom_model_2(frame, imgsz=640, conf=0.5, iou=0.45, device=0, verbose=False, classes = [3, 4, 5])
    results_custom_2[0].names = custom_names_2

    results_custom_3 = custom_model_3(frame, imgsz=640, conf=0.5, iou=0.45, device=0, verbose=False, classes = [2, 3, 4, 5, 6])
    results_custom_3[0].names = custom_names_3

    # 결과 통합 및 정렬
    all_detections = []
    all_detections.extend(get_detections(results_custom, custom_names_1))
    all_detections.extend(get_detections(results_custom_2, custom_names_2))
    all_detections.extend(get_detections(results_custom_3, custom_names_3))
    all_detections.extend(get_detections(results_coco, coco_names))
    all_detections.sort(key=lambda x: x['conf'], reverse=True)

    # 방향별 그룹화
    direction_groups = {"왼쪽에": set(), "정면": set(), "오른쪽에": set()}
    for det in all_detections:
        class_name = det['class_name']
        x1, _, x2, _ = det['box']
        x_center = (x1 + x2) / 2
        
        if x_center < CAMW / 3:
            direction = "왼쪽에"
        elif x_center > (CAMW * 2) / 3:
            direction = "오른쪽에"
        else:
            direction = "정면"
        direction_groups[direction].add(class_name)

    # 문자열 결합
    spoken_segments = []
    for direction, items in direction_groups.items():
        if items:  
            items_str = ", ".join(list(items))
            spoken_segments.append(f"{direction} {items_str}")

    current_spoken_text = ". ".join(spoken_segments)

    # ------------------------------------------
    # [핵심 수정] Ollama의 상태 유무에 따른 동기화 제어
    # ------------------------------------------
    if current_spoken_text:
        time_elapsed = current_time - last_spoken_time
        is_new_situation = (current_spoken_text != last_spoken_text)
        
        # 1. Ollama가 현재 쉬고 있는 상태(not ollama_is_busy)인지 먼저 검사합니다.
        if not ollama_is_busy:
            # 2. 그 상태에서 새로운 상황이거나 혹은 동일 상황에서 쿨다운이 지났다면 요청 진입
            if is_new_situation or (time_elapsed > ALERT_COOLDOWN):
                with lock:
                    ollama_is_busy = True # 즉시 문을 잠가서 다른 프레임 데이터 유입 차단
                
                ollama_queue.put(current_spoken_text)
                
                last_spoken_text = current_spoken_text
                last_spoken_time = current_time

    # 시각화
    frame = results_custom[0].plot(img=frame)
    frame = results_coco[0].plot(img=frame)
    frame = results_custom_2[0].plot(img=frame)
    frame = results_custom_3[0].plot(img=frame)

    # FPS 계산
    cur_time = time.time()
    dt = cur_time - prev_time
    if dt > 0: fps = 1.0 / dt
    prev_time = cur_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 