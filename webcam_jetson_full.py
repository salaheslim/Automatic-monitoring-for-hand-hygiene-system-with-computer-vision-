import cv2
import numpy as np
import xgboost as xgb
import time
from collections import deque
import mediapipe as mp
from sklearn.preprocessing import StandardScaler

# Load Jetson personalised model
booster = xgb.Booster()
booster.load_model('xgb_model_jetson.json')

# Load Jetson scaler
scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean_jetson.npy')
scaler.scale_ = np.load('scaler_scale_jetson.npy')
scaler.var_ = scaler.scale_ ** 2
scaler.n_features_in_ = 126
print("Jetson model and scaler loaded successfully")

STEPS = {
    0: "Step 1: Palm to palm",
    1: "Step 2: Right over left",
    2: "Step 3: Fingers interlaced",
    3: "Step 4: Backs of fingers",
    4: "Step 5: Rotational thumbs",
    5: "Step 6: Fingertip rubbing",
    6: "Step 7: Wrists"
}

CONFIRMATION_TIME = 1.5
VOTE_THRESHOLD = 0.65
VOTE_WINDOW = 5

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Session state
current_step_idx = 0
step_results = []
step_start_time = None
session_start = time.time()
session_complete = False
vote_buffer = deque(maxlen=VOTE_WINDOW)
confirm_start = None
frame_count = 0
fps_start = time.time()
fps = 0

# Colours
DARK_BG = (30, 30, 30)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
RED = (0, 0, 255)
TEAL = (255, 200, 0)

print("WHO Hand Hygiene Detection — Jetson Nano")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    features = np.zeros(126)
    hands_detected = 0

    if results.multi_hand_landmarks:
        hands_detected = len(results.multi_hand_landmarks)
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255,100,0), thickness=2)
            )
            lm = hand_landmarks.landmark
            offset = 63 if idx == 1 else 0
            for i, l in enumerate(lm):
                features[offset + i*3] = l.x
                features[offset + i*3+1] = l.y
                features[offset + i*3+2] = l.z

    # Prediction
    pred = current_step_idx
    confidence = 0.0
    if hands_detected > 0 and not session_complete:
        scaled = scaler.transform(features.reshape(1, -1))
        dmatrix = xgb.DMatrix(scaled)
        probs = booster.predict(dmatrix)[0]
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
        vote_buffer.append(pred)

    # Voting
    smooth_pred = 0
    if vote_buffer:
        counts = {}
        for v in vote_buffer:
            counts[v] = counts.get(v, 0) + 1
        smooth_pred = max(counts, key=counts.get)
        vote_conf = counts[smooth_pred] / len(vote_buffer)
    else:
        vote_conf = 0

    # Step confirmation logic
    if not session_complete and current_step_idx < 7:
        target_step = current_step_idx
        if smooth_pred == target_step and vote_conf >= VOTE_THRESHOLD and hands_detected > 0:
            if confirm_start is None:
                confirm_start = time.time()
                step_start_time = time.time()
            elapsed_confirm = time.time() - confirm_start
            if elapsed_confirm >= CONFIRMATION_TIME:
                scan_time = time.time() - step_start_time
                step_results.append({
                    'step': target_step,
                    'name': STEPS[target_step],
                    'scan_time': scan_time,
                    'status': 'Done'
                })
                print(f"✅ {STEPS[target_step]} — {scan_time:.2f}s")
                current_step_idx += 1
                confirm_start = None
                step_start_time = None
                if current_step_idx >= 7:
                    session_complete = True
                    print("\n🎉 Session Complete — 7/7 steps!")
        else:
            confirm_start = None

    # FPS
    frame_count += 1
    elapsed = time.time() - fps_start
    if elapsed > 0:
        fps = frame_count / elapsed

    # ── BUILD UI ──────────────────────────────────────────────────────
    # Create right panel
    panel_w = 380
    total_w = 640 + panel_w
    canvas = np.zeros((480, total_w, 3), dtype=np.uint8)
    canvas[:, :640] = frame

    # Dark panel background
    canvas[:, 640:] = DARK_BG

    # Title
    title = "WHO Hand Hygiene | Jetson Nano"
    cv2.putText(canvas, title, (648, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEAL, 2)
    cv2.putText(canvas, f"XGBoost GPU | Hands: {hands_detected}/2",
                (648, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)
    cv2.putText(canvas, f"FPS: {fps:.1f} | Step detection + scan time",
                (648, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)

    # Divider
    cv2.line(canvas, (648, 78), (1010, 78), (70,70,70), 1)

    # Step table header
    cv2.putText(canvas, "Step", (648, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)
    cv2.putText(canvas, "Status", (830, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)
    cv2.putText(canvas, "Time", (930, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)
    cv2.line(canvas, (648, 105), (1010, 105), (70,70,70), 1)

    # Step rows
    for s in range(7):
        y = 125 + s * 42
        row_color = (45, 45, 45) if s % 2 == 0 else (38, 38, 38)
        cv2.rectangle(canvas, (644, y-18), (1015, y+20), row_color, -1)

        # Step name
        step_short = STEPS[s][:22]
        if s < len(step_results):
            # Completed
            cv2.putText(canvas, step_short, (648, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
            # Green OK badge
            cv2.rectangle(canvas, (825, y-14), (885, y+8), (0,120,0), -1)
            cv2.putText(canvas, "OK", (835, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
            cv2.putText(canvas, "Done", (895, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1)
            scan_t = step_results[s]['scan_time']
            cv2.putText(canvas, f"{scan_t:.1f}s", (950, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1)
        elif s == current_step_idx and not session_complete:
            # Current step — highlight
            cv2.rectangle(canvas, (644, y-18), (1015, y+20), (0,60,0), -1)
            cv2.putText(canvas, step_short, (648, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 2)
            # Confirmation progress
            if confirm_start is not None:
                prog = min((time.time()-confirm_start)/CONFIRMATION_TIME, 1.0)
                bar_w = int(prog * 120)
                cv2.rectangle(canvas, (825, y-8), (945, y+5), (50,50,50), -1)
                cv2.rectangle(canvas, (825, y-8), (825+bar_w, y+5), GREEN, -1)
                cv2.putText(canvas, f"{prog*100:.0f}%", (950, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1)
            else:
                cv2.putText(canvas, "Waiting...", (825, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, ORANGE, 1)
        else:
            # Pending
            cv2.putText(canvas, step_short, (648, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
            cv2.putText(canvas, "---", (895, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
            cv2.putText(canvas, "-.--s", (950, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)

    # Divider before summary
    cv2.line(canvas, (648, 425), (1010, 425), (70,70,70), 1)

    # Session summary
    steps_done = len(step_results)
    compliance = (steps_done / 7) * 100
    avg_time = np.mean([r['scan_time'] for r in step_results]) if step_results else 0
    total_time = time.time() - session_start

    cv2.putText(canvas, f"Steps: {steps_done}/7", (648, 445),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(canvas, f"Compliance: {compliance:.0f}%", (648, 462),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN if compliance == 100 else ORANGE, 1)
    cv2.putText(canvas, f"Avg: {avg_time:.1f}s", (790, 445),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(canvas, f"Total: {total_time:.1f}s", (790, 462),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    if session_complete:
        cv2.rectangle(canvas, (644, 0), (1015, 480), GREEN, 3)
        cv2.putText(canvas, "SESSION COMPLETE!", (660, 478),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

    cv2.imshow("WHO Hand Hygiene | Jetson Nano", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final report
print("\n" + "="*50)
print("JETSON NANO — SESSION REPORT")
print("="*50)
for r in step_results:
    print(f"  {r['name']}: {r['scan_time']:.2f}s")
if step_results:
    avg = np.mean([r['scan_time'] for r in step_results])
    total = time.time() - session_start
    print(f"\nSteps completed: {len(step_results)}/7")
    print(f"Compliance: {(len(step_results)/7)*100:.0f}%")
    print(f"Average scan time: {avg:.2f}s")
    print(f"Total session time: {total:.1f}s")
    print(f"Average FPS: {fps:.1f}")
    print(f"Device: Jetson Nano 4GB (GPU accelerated)")
print("="*50)
