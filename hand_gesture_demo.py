import cv2
import mediapipe as mp
import pyautogui
import time

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

last_gesture = ""
last_action_time = 0
COOLDOWN = 1.0  # seconds


def detect_gesture(lm):
    wrist = lm[0]

    thumb_tip = lm[4]
    index_tip = lm[8]
    middle_tip = lm[12]
    ring_tip = lm[16]
    pinky_tip = lm[20]

    index_mcp = lm[5]
    middle_mcp = lm[9]
    ring_mcp = lm[13]
    pinky_mcp = lm[17]

    # Fingers open state
    fingers = [
        index_tip[1] < index_mcp[1],
        middle_tip[1] < middle_mcp[1],
        ring_tip[1] < ring_mcp[1],
        pinky_tip[1] < pinky_mcp[1]
    ]

    # ----- Gesture Logic -----

    # Thumbs Up (only thumb up)
    if thumb_tip[1] < wrist[1] and fingers == [False, False, False, False]:
        return "Thumbs Up"

    # Thumbs Down (only thumb down)
    if thumb_tip[1] > wrist[1] and fingers == [False, False, False, False]:
        return "Thumbs Down"

    # Open Palm
    if all(fingers):
        return "Open Palm"

    # Fist
    if not any(fingers):
        return "Fist"

    return "Unknown"


def perform_action(gesture):
    global last_action_time

    current_time = time.time()
    if current_time - last_action_time < COOLDOWN:
        return

    if gesture == "Open Palm":
        pyautogui.press("playpause")

    elif gesture == "Fist":
        pyautogui.press("playpause")

    elif gesture == "Thumbs Up":
        pyautogui.press("volumeup")

    elif gesture == "Thumbs Down":
        pyautogui.press("volumedown")

    last_action_time = current_time


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm_list = []
            h, w, _ = frame.shape

            for lm in hand.landmark:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            gesture = detect_gesture(lm_list)

            if gesture != last_gesture:
                perform_action(gesture)
                last_gesture = gesture

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Gesture Media Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
