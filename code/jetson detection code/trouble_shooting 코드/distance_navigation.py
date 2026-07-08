import cv2
import time
import threading
import queue
import ollama
from ultralytics import YOLO
import pyttsx3

# ==========================================
# 1. 초기 설정 (해상도 및 쿨다운)
# ==========================================
CAMW = 640
CAMH = 480
NORMAL_ALERT_INTERVAL = 3.0  # 일반 안내 주기
URGENT_COOLDOWN = 5.0        # 긴급 안내 쿨다운

audio_queue = queue.Queue()
ollama_queue = queue.Queue()
ollama_is_busy = False
lock = threading.Lock()

# ==========================================
# 2. 오디오 및 Ollama 쓰레드
# ==========================================
def audio_worker():
    """텍스트를 음성으로 변환하고 즉시 재생"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    
    # 한국어 음성 강제 적용
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'ko' in voice.id.lower() or 'korean' in voice.name.lower() or 'korea' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
            
    while True:
        message = audio_queue.get()
        print(f"🔊 [스피커 출력]: {message}")
        try:
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print(f"❌ [오디오 에러]: {e}")
        finally:
            audio_queue.task_done()

ollama_history = [
    {
        'role': 'system', 
        'content': '당신은 시각장애인을 위한 이동 보조 AI입니다. 입력되는 사물 정보를 바탕으로, 보행자가 주의해야 할 핵심 요약 안내를 한국어 한 문장(15자 이내)으로 매우 짧고 직관적으로 답하세요.'
    }
]

def ollama_worker():
    """자연어 처리 전담 (Qwen2 최적화 적용)"""
    global ollama_history, ollama_is_busy
    while True:
        text_to_analyze = ollama_queue.get()
        print(f"\n🧠 [Ollama 분석 시작]: {text_to_analyze}")
        
        ollama_history.append({'role': 'user', 'content': f"현재 상태: {text_to_analyze}"})
        
        try:
            stream = ollama.chat(
                model='qwen2:0.5b', 
                messages=ollama_history,
                stream=True,
                options={
                    'num_predict': 20,
                    'temperature': 0.1,
                    'top_k': 10,
                },
            )
            
            ai_response = ""
            final_response = None
            
            for chunk in stream:
                ai_response += chunk['message']['content']
                if chunk.get('done'):
                    final_response = chunk

            # 성능 출력
            if final_response and 'eval_count' in final_response:
                eval_count = final_response['eval_count']
                eval_duration = final_response['eval_duration']
                speed = eval_count / (eval_duration / 1e9)
                print(f"📊 [성능 분석] 속도: {speed:.2f} tokens/s | 생성된 토큰: {eval_count}")
                
            ollama_history.append({'role': 'assistant', 'content': ai_response})
            if len(ollama_history) > 6:
                ollama_history = [ollama_history[0]] + ollama_history[-4:]
                
            while not audio_queue.empty():
                try: audio_queue.get_nowait()
                except queue.Empty: break
            audio_queue.put(ai_response)
            
        except Exception as e:
            print(f"❌ [Ollama 에러]: {e}")
            if len(ollama_history) > 1: ollama_history.pop()
        finally:
            with lock:
                ollama_is_busy = False
            ollama_queue.task_done()

# 쓰레드 실행
threading.Thread(target=audio_worker, daemon=True).start()
threading.Thread(target=ollama_worker, daemon=True).start()

# ==========================================
# 3. 위치 및 거리 연산 함수
# ==========================================
def get_direction(x_center, img_width):
    left_threshold = img_width * 0.33
    right_threshold = img_width * 0.66
    if x_center < left_threshold: return "왼쪽에"
    elif x_center > right_threshold: return "오른쪽에"
    else: return "정면에"

def get_distance_category(label, h, y_center, img_h):
    """픽셀 비율 기반 3단계 거리 추정"""
    if label in ['빨간불', '초록불']:
        if h > img_h * 0.30: return "매우 가까움"
        elif h > img_h * 0.15: return "가까움"
        else: return "멀음"
    else:
        y_max = y_center + (h / 2)
        if y_max > img_h * 0.85: return "매우 가까움"
        elif y_max > img_h * 0.60: return "가까움"
        else: return "멀음"

# ==========================================
# 4. 메인 카메라 모델 로드
# ==========================================
model = YOLO("navigation-1_yolo11n_6402.engine", task="detect")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMH)

custom_names = {2: '횡단보도', 3: '내려가는계단', 4: '초록불', 5: '빨간불', 6: '올라가는계단'}

last_spoken_time = 0.0
last_urgent_time = 0.0
prev_time = time.time()

# ==========================================
# 5. 영상 처리 루프
# ==========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    current_time = time.time()
    results = model(frame, imgsz=640, conf=0.5, iou=0.45, device=0, verbose=False, classes=[2, 3, 4, 5, 6])
    results[0].names = custom_names 

    detected_items = []
    urgent_flags = []
    
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        x_center, y_center, w, h = box.xywh[0].tolist()
        
        direction = get_direction(x_center, CAMW)
        label = custom_names[class_id]
        
        # 거리 측정 및 텍스트 변환
        dist_category = get_distance_category(label, h, y_center, CAMH)
        dist_text = "아주 가까운" if dist_category == "매우 가까움" else "가까운" if dist_category == "가까움" else "멀리 있는"
            
        detected_items.append(f"{direction} {dist_text} {label}")
        
        # 위험 객체가 근접했을 때만 긴급 배열에 추가
        if label in ['빨간불', '초록불', '내려가는계단'] and dist_category in ["매우 가까움", "가까움"]:
            urgent_flags.append((label, dist_category))

    current_status = ", ".join(list(set(detected_items)))

    # 긴급 vs 일반 상황 분기 (LLM 호출 여부 결정)
    if urgent_flags:
        if current_time - last_urgent_time > URGENT_COOLDOWN:
            for item, dist_cat in urgent_flags:
                prefix = "주의! 바로 앞에" if dist_cat == "매우 가까움" else "전방에"
                if item == '빨간불': audio_queue.put(f"{prefix} 빨간불입니다. 정지하세요.")
                elif item == '초록불': audio_queue.put(f"{prefix} 초록불입니다. 건너세요.")
                elif item == '내려가는계단': audio_queue.put(f"{prefix} 내려가는 계단이 있습니다.")
            last_urgent_time = current_time

    elif current_status:
        # 긴급 상황이 아닐 때는 무조건 LLM(Qwen2)에게 텍스트 생성을 맡김
        if not ollama_is_busy and (current_time - last_spoken_time > NORMAL_ALERT_INTERVAL):
            with lock:
                ollama_is_busy = True
            ollama_queue.put(current_status)
            last_spoken_time = current_time

    # FPS 출력 및 화면 표시
    frame = results[0].plot()
    cur_time = time.time()
    dt = cur_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0
    prev_time = cur_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()