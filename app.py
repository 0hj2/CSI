# app.py
# -*- coding: utf-8 -*-
import os
import time
import socket
import threading
from queue import Queue, Empty
from flask import Flask, jsonify, render_template
import keras
import numpy as np
import io
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import json

# =========================
# 0. 전역 변수 / 환경 변수
# =========================
HISTORY_FILE = "activity_history.json"
last_reset_date = time.strftime("%Y-%m-%d")
BUFFER_SIZE = 20

csi_data_queue = Queue(maxsize=5)
temp_buffer = []

user_state = "없음"
state_start_time = time.time()

activity_summary = {"앉기": 0.0, "서기": 0.0, "걷기": 0.0}

last_prediction = {"state": "없음", "time": 0}

last_guidance_time = {"앉기": 0, "서기": 0, "걷기": 0}
GUIDANCE_INTERVAL_MIN = 0.0833  # 30분 #테스트용


# Gemini 설정
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or ""
GEMINI_MODEL = None
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
    print("[INFO] Gemini API 활성화")
else:
    print("[WARNING] GEMINI_API_KEY가 설정되지 않아 건강 가이드 기능 비활성화됩니다.")

# =========================
# 1. 모델 로드
# =========================
MODEL_PATH = r"C:\Users\0hyej\OneDrive\바탕 화면\trained10.keras"
try:
    LOADED_MODEL = keras.models.load_model(MODEL_PATH)
    print("[INFO] 모델 로드 성공")
except Exception as e:
    print(f"[ERROR] 모델 로드 실패: {e}")
    LOADED_MODEL = None

MODEL_CLASSES = ['losdown', 'losup', 'loswalk', 'losN']
USER_STATE_MAP = {"losdown": "앉기", "losup": "서기", "loswalk": "걷기", "losN": "없음"}

# =========================
# 2. Flask 초기화
# =========================
app = Flask(__name__, template_folder="templates")

# =========================
# 3. 일일 리셋
# =========================
def reset_if_new_day():
    global activity_summary, last_reset_date
    today = time.strftime("%Y-%m-%d")
    if today != last_reset_date:
        record = {"date": last_reset_date, **activity_summary}
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
        history.append(record)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        activity_summary = {"앉기": 0.0, "서기": 0.0, "걷기": 0.0}
        last_reset_date = today
        print(f"[INFO] 일일 리셋 완료: {last_reset_date}")

# =========================
# 4. UDP 수신 스레드
# =========================
def udp_listener():
    global temp_buffer
    HOST, PORT = "0.0.0.0", 5500
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    print(f"[UDP] {HOST}:{PORT} 수신 대기 중...")
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            amplitudes = np.frombuffer(data, dtype=np.float32)
            if amplitudes.size != 64:
                continue
            temp_buffer.append(amplitudes)

            while len(temp_buffer) >= BUFFER_SIZE:
                csi_matrix = np.vstack(temp_buffer[:BUFFER_SIZE])
                temp_buffer = temp_buffer[BUFFER_SIZE:]
                if not csi_data_queue.empty():
                    try: csi_data_queue.get_nowait()
                    except Empty: pass
                csi_data_queue.put(csi_matrix)
        except Exception as e:
            print(f"[UDP ERROR] {e}")
            break
    sock.close()

# =========================
# 5. 전처리
# =========================
def preprocess_csi(raw_csi_data):
    try:
        amplitudes = np.abs(raw_csi_data)
        db_values = 20 * np.log10(np.where(amplitudes>0, amplitudes, np.nan))
        db_values = np.where(db_values>70, 55.94, db_values)
        data = pd.DataFrame(db_values)

        plt.figure(figsize=(2.24,2.24), dpi=100)
        sns.heatmap(data, cmap="coolwarm", cbar=False, xticklabels=False, yticklabels=False)
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)/255.0
        img.close()
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"[PREPROCESS ERROR] {e}")
        return None

# =========================
# 6. 상태 분류
# =========================
def classify_user_state():
    global last_prediction
    if csi_data_queue.empty():
        if time.time() - last_prediction["time"] < 10:
            return last_prediction["state"]
        else:
            return "없음"

    try:
        csi_data = csi_data_queue.get_nowait()
        processed = preprocess_csi(csi_data)
        if processed is None: return last_prediction["state"]
        pred = LOADED_MODEL.predict(processed)
        pred_index = np.argmax(pred)
        model_class = MODEL_CLASSES[pred_index]
        new_state = USER_STATE_MAP.get(model_class, "없음")
        last_prediction = {"state": new_state, "time": time.time()}
        return new_state
    except Exception as e:
        print(f"[MODEL ERROR] {e}")
        return last_prediction["state"]

# =========================
# 7. Gemini 가이드
# =========================
def get_gemini_guidance(state, duration_minutes):
    if GEMINI_MODEL is None:
        return "건강 가이드를 생성할 수 없습니다."
    safe_duration = max(30, round(duration_minutes))
    prompt = ""
    if state == "앉기":
        prompt = f"사용자는 {safe_duration}분 동안 앉아있습니다. 간단한 스트레칭 방법을 세문장으로 알려줘."
    elif state == "서기":
        prompt = f"사용자는 {safe_duration}분 동안 서있습니다. 다리 건강 가이드를 세문장으로 알려줘."
    elif state == "걷기":
        prompt = f"사용자는 {safe_duration}분 동안 걷고 있습니다. 걷는 것에 대한 간단한 건강 가이드 세문장으로 알려줘."
    
    print(f"[DEBUG] Gemini prompt: {prompt}")  # ✅ 추가

    try:
        response = GEMINI_MODEL.generate_content(prompt)
        print(f"[DEBUG] Gemini response: '{response.text}'")  # ✅ 추가
        if response.text.strip(): return response.text
        return "건강 가이드를 생성할 수 없습니다."
    except Exception as e:
        print(f"[Gemini ERROR] {e}")
        return "건강 가이드를 생성할 수 없습니다."


# =========================
# 8. Flask 라우트
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    reset_if_new_day()
    global user_state, state_start_time

    now = time.time()
    new_state = classify_user_state()

    # 상태 변경 감지 → 타이머 초기화
    if new_state != user_state:
        elapsed = now - state_start_time
        if user_state in activity_summary:
            activity_summary[user_state] += elapsed / 60.0
        user_state = new_state
        state_start_time = now
        # 상태 바뀌면 마지막 Gemini 호출 시간 초기화
        last_guidance_time[user_state] = 0

    duration_sec = now - state_start_time
    duration_min = duration_sec / 60

    guidance = f"{user_state} 상태 유지 중..."

    # 5초 이상 유지 시 Gemini 호출 (같은 상태에서는 한 번만)
    if user_state in activity_summary and duration_sec >= 5: #테스트용
        last_call = last_guidance_time.get(user_state, 0)
        if last_call == 0:  # 아직 한 번도 호출하지 않았을 때만
            guidance = get_gemini_guidance(user_state, duration_min)
            last_guidance_time[user_state] = now

    return jsonify({
        "status": user_state,
        "duration": f"{int(duration_sec // 60)}분 {int(duration_sec % 60)}초",
        "duration_seconds": int(duration_sec),
        "guidance": guidance
    })



@app.route("/daily_summary")
def daily_summary():
    reset_if_new_day()
    now = time.time()
    elapsed = now - state_start_time

    summary_copy = activity_summary.copy()
    if user_state in summary_copy:
        summary_copy[user_state] += elapsed / 60.0

    total_time = sum(summary_copy.values())
    goal_percentage = min((total_time/120)*100, 100)

    return jsonify({
        "total_time": total_time,
        "sitting_time": summary_copy["앉기"],
        "standing_time": summary_copy["서기"],
        "walking_time": summary_copy["걷기"],
        "goal_percentage": goal_percentage
    })

@app.route("/weekly_chart")
def weekly_chart():
    import datetime
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else: history = []

    weekday_totals = [0]*7
    weekday_counts = [0]*7
    for rec in history[-30:]:
        dt = datetime.datetime.strptime(rec["date"], "%Y-%m-%d")
        weekday = dt.weekday()
        total = rec.get("앉기",0)+rec.get("서기",0)+rec.get("걷기",0)
        weekday_totals[weekday] += total
        weekday_counts[weekday] += 1
    weekday_avg = [weekday_totals[i]/weekday_counts[i] if weekday_counts[i]>0 else 0 for i in range(7)]
    labels = ["월","화","수","목","금","토","일"]
    return jsonify({"labels": labels, "data": weekday_avg})


# =========================
# 9. Flask 실행
# =========================
if __name__ == "__main__":
    threading.Thread(target=udp_listener, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
