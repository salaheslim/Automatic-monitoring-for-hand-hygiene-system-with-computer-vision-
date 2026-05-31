import cv2
import numpy as np
import mediapipe as mp
import os
from wrist_normalise import wrist_anchor_normalise

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs('jetson_data_normalised', exist_ok=True)

cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

STEPS = {
    0: "Step 1: Palm to palm",
    1: "Step 2: Right over left",
    2: "Step 3: Fingers interlaced",
    3: "Step 4: Backs of fingers",
    4: "Step 5: Rotational thumbs",
    5: "Step 6: Fingertip rubbing",
    6: "Step 7: Wrists"
}

FRAMES_PER_STEP = 300

print("="*50)
print("NORMALISED DATA COLLECTION")
print("="*50)
print(f"Collecting {FRAMES_PER_STEP} frames per step")
print("Wrist-anchor normalisation ENABLED")
print("Click on camera window first")
print("Press 0-6 to record step")
print("Press R to reset current step")
print("Press Q to quit")
print("="*50)

current_step = None
step_data = {i: [] for i in range(7)}
recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    features = np.zeros(126)
    hands_found = False

    if results.multi_hand_landmarks:
        hands_found = True
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
            )
            lm = hand_landmarks.landmark
            offset = 63 if idx == 1 else 0
            for i, l in enumerate(lm):
                features[offset + i*3]   = l.x
                features[offset + i*3+1] = l.y
                features[offset + i*3+2] = l.z

        # Apply wrist-anchor normalisation
        features = wrist_anchor_normalise(features)

    if current_step is not None and hands_found and recording:
        step_data[current_step].append(features.copy())
        count = len(step_data[current_step])

        # Progress bar
        progress = int((count / FRAMES_PER_STEP) * 400)
        cv2.rectangle(frame, (10,440), (410,465), (50,50,50), -1)
        cv2.rectangle(frame, (10,440), (10+progress,465), (0,255,0), -1)
        cv2.putText(frame, f"{count}/{FRAMES_PER_STEP}",
                   (420,460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"RECORDING: {STEPS[current_step]}",
                   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
        cv2.putText(frame, "NORMALISED",
                   (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)

        if count >= FRAMES_PER_STEP:
            filename = f"jetson_data_normalised/step{current_step}_norm.npy"
            np.save(filename, np.array(step_data[current_step]))
            print(f"Saved {STEPS[current_step]} — {count} frames")
            current_step = None
            recording = False

    elif current_step is not None and not recording:
        cv2.putText(frame, f"GET READY: {STEPS[current_step]}",
                   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2)
        if hands_found:
            recording = True
    else:
        cv2.putText(frame, "Press 0-6 to record step",
                   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)

    # Show progress
    for s in range(7):
        count = len(step_data[s])
        color = (0,255,0) if count >= FRAMES_PER_STEP else (0,165,255)
        status = "DONE" if count >= FRAMES_PER_STEP else f"{count}/{FRAMES_PER_STEP}"
        cv2.putText(frame, f"S{s+1}:{status}",
                   (480,60+s*30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    hand_color = (0,255,0) if hands_found else (0,0,255)
    cv2.putText(frame,
                f"Hands:{len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}",
                (10,410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

    cv2.imshow("Normalised Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and current_step is not None:
        step_data[current_step] = []
        recording = False
        print(f"Reset Step {current_step+1}")
    elif key in [ord('0'),ord('1'),ord('2'),ord('3'),
                 ord('4'),ord('5'),ord('6')]:
        current_step = int(chr(key))
        recording = False
        print(f"Get ready for {STEPS[current_step]}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("COLLECTION SUMMARY")
print("="*50)
for s in range(7):
    count = len(step_data[s])
    status = "DONE" if count >= FRAMES_PER_STEP else f"{count} frames"
    print(f"  {STEPS[s]}: {status}")
print("="*50)
