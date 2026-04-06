import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            index_x = int(landmarks[8].x * screen_width)
            index_y = int(landmarks[8].y * screen_height)

            thumb_x = int(landmarks[4].x * screen_width)
            thumb_y = int(landmarks[4].y * screen_height)

            # Move Mouse
            pyautogui.moveTo(index_x, index_y)

            # Click Gesture (thumb + index close)
            distance = np.hypot(index_x - thumb_x, index_y - thumb_y)

            if distance < 40:
                pyautogui.click()
                pyautogui.sleep(0.2)

    cv2.imshow("Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()