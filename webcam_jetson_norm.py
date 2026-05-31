import cv2
import numpy as np
import xgboost as xgb
import time
from collections import deque
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
from wrist_normalise import wrist_anchor_normalise

# Load normalised model
booster = xgb.Booster()
booster.load_model('xgb_model_norm.json')

scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean_norm.npy')
scaler.scale_ = np.load('scaler_scale_norm.npy')
scaler.var_ = scaler.scale_ ** 2
scaler.n_features_in_ = 126
print("Normalised model loaded successfully")

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
session_start = time.time()
session_complete = False
vote_buffer = deque(maxlen=5)
confirm_start = None
step_start = None
frame_count = 0
fps_start = time.time()
fps = 0

DARK_BG = (30,30,30)
GREEN = (0,255,0)
ORANGE = (0,165,255)
WHITE = (255,255,255)
GRAY = (150,150,150)
TEAL = (255,200,0)

print("WHO Hand Hygiene — Wrist-Anchor Normalisation")
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
                features[offset + i*3]   = l.x
                features[offset + i*3+1] = l.y
                features[offset + i*3+2] = l.z

        # Apply wrist-anchor normalisation
        features = wrist_anchor_normalise(features)

    pred = current_step_idx
    confidence = 0.0
    if hands_detected > 0 and not session_complete:
        scaled = scaler.transform(features.reshape(1,-1))
        dmatrix = xgb.DMatrix(scaled)
        probs = booster.predict(dmatrix)[0]
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
        vote_buffer.append(pred)

    smooth_pred = 0
    vote_conf = 0
    if vote_buffer:
        counts = {}
        for v in vote_buffer:
            counts[v] = counts.get(v,0) + 1
        smooth_pred = max(counts, key=counts.get)
        vote_conf = counts[smooth_pred] / len(vote_buffer)

    if not session_complete and current_step_idx < 7:
        if smooth_pred == current_step_idx and \
           vote_conf >= VOTE_THRESHOLD and hands_detected > 0:
            if confirm_start is None:
                confirm_start = time.time()
                step_start = time.time()
            if time.time() - confirm_start >= CONFIRMATION_TIME:
                scan_time = time.time() - step_start
                step_results.append({
                    'step': current_step_idx,
                    'name': STEPS[current_step_idx],
                    'scan_time': scan_time
                })
                print(f"Step {current_step_idx+1}: {scan_time:.2f}s")
                current_step_idx += 1
                confirm_start = None
                if current_step_idx >= 7:
                    session_complete = True
                    print("Session Complete!")
        else:
            confirm_start = None

    frame_count += 1
    elapsed = time.time() - fps_start
    if elapsed > 0:
        fps = frame_count / elapsed

    # Build UI
    panel_w = 380
    canvas = np.zeros((480, 640+panel_w, 3), dtype=np.uint8)
    canvas[:,:640] = frame
    canvas[:,640:] = DARK_BG

    cv2.putText(canvas, "WHO Hand Hygiene | Wrist-Anchor Norm",
                (648,25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEAL, 2)
    cv2.putText(canvas, f"FPS:{fps:.1f} | Hands:{hands_detected} | Normalised",
                (648,48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
    cv2.line(canvas, (648,58), (1010,58), (70,70,70), 1)

    cv2.putText(canvas, "Step", (648,78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
    cv2.putText(canvas, "Status", (830,78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
    cv2.putText(canvas, "Time", (930,78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
    cv2.line(canvas, (648,85), (1010,85), (70,70,70), 1)

    for s in range(7):
        y = 105 + s*42
        rc = (45,45,45) if s%2==0 else (38,38,38)
        cv2.rectangle(canvas, (644,y-18), (1015,y+20), rc, -1)
        sname = STEPS[s][:22]
        if s < len(step_results):
            cv2.putText(canvas, sname, (648,y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
            cv2.rectangle(canvas, (825,y-14), (885,y+8), (0,120,0), -1)
            cv2.putText(canvas, "OK", (835,y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
            cv2.putText(canvas, "Done", (895,y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1)
            cv2.putText(canvas, f"{step_results[s]['scan_time']:.1f}s",
                       (950,y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1)
        elif s == current_step_idx and not session_complete:
            cv2.rectangle(canvas, (644,y-18), (1015,y+20), (0,60,0), -1)
            cv2.putText(canvas, sname, (648,y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 2)
            if confirm_start is not None:
                prog = min((time.time()-confirm_start)/CONFIRMATION_TIME,1.0)
                bw = int(prog*120)
                cv2.rectangle(canvas, (825,y-8), (945,y+5), (50,50,50), -1)
                cv2.rectangle(canvas, (825,y-8), (825+bw,y+5), GREEN, -1)
                cv2.putText(canvas, f"{prog*100:.0f}%",
                           (950,y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1)
            else:
                cv2.putText(canvas, "Waiting...",
                           (825,y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ORANGE, 1)
        else:
            cv2.putText(canvas, sname, (648,y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)
            cv2.putText(canvas, "---", (895,y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1)

    cv2.line(canvas, (648,405), (1010,405), (70,70,70), 1)
    steps_done = len(step_results)
    avg_time = np.mean([r['scan_time'] for r in step_results]) \
               if step_results else 0
    total_time = time.time() - session_start

    cv2.putText(canvas,
                f"Steps:{steps_done}/7  Compliance:{(steps_done/7)*100:.0f}%",
                (648,425), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)
    cv2.putText(canvas,
                f"Avg:{avg_time:.1f}s  Total:{total_time:.1f}s",
                (648,445), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)
    cv2.putText(canvas, "Wrist-Anchor Normalisation Active",
                (648,465), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEAL, 1)

    if session_complete:
        cv2.rectangle(canvas, (644,0), (1015,480), GREEN, 3)

    cv2.imshow("WHO Hand Hygiene | Normalised", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("NORMALISED MODEL — SESSION REPORT")
print("="*50)
for r in step_results:
    print(f"  {r['name']}: {r['scan_time']:.2f}s")
if step_results:
    avg = np.mean([r['scan_time'] for r in step_results])
    print(f"\nSteps: {len(step_results)}/7")
    print(f"Compliance: {(len(step_results)/7)*100:.0f}%")
    print(f"Average scan time: {avg:.2f}s")
    print(f"FPS: {fps:.1f}")
    print(f"Normalisation: Wrist-anchor applied")
print("="*50)
