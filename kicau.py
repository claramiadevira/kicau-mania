#set up envi
import cv2
import mediapipe as mp
import math
import pygame
import numpy as np
import imageio

#buat musik
pygame.mixer.init()
pygame.mixer.music.load("kicau.mp3")

#buat generate gif nya
gif1 = imageio.mimread("cat.gif")
gif1_frames = []
for frame in gif1:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (180, 180))
    gif1_frames.append(frame)

gif1_index = 0

#ini gif windah

gif2 = imageio.mimread("windah.gif")
gif2_frames = []
for frame in gif2:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (180, 180))
    gif2_frames.append(frame)

gif2_index = 0

#setup buat mediapipe nya
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

#akses webcam

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#resolusi direndahin
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

#efek biar lucu
triggered = False
frame_count = 0

while True:

    success, img = cap.read()

    if not success:
        break

    #mirror webcam nya
    img = cv2.flip(img, 1)

    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    nose_x, nose_y = None, None
    finger_x, finger_y = None, None
    forehead_x, forehead_y = None, None

    #detect face

    if face_results.multi_face_landmarks:

        for face_landmarks in face_results.multi_face_landmarks:

            #hidung
            nose = face_landmarks.landmark[1]

            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            #kening
            forehead = face_landmarks.landmark[10]
            forehead_x = int(forehead.x * w)
            forehead_y = int(forehead.y * h)

    #hand detect

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            index_finger = hand_landmarks.landmark[8]
            finger_x = int(index_finger.x * w)
            finger_y = int(index_finger.y * h)

    #pegang hidung
    if (
        nose_x is not None and
        finger_x is not None
    ):

        distance = math.hypot(
            finger_x - nose_x,
            finger_y - nose_y
        )

        if distance < 20:

            if not triggered:

                triggered = True

                pygame.mixer.music.play(-1)

        else:

            triggered = False

            pygame.mixer.music.stop()

    if triggered:

        frame_count += 1
        glow = cv2.GaussianBlur(img, (0, 0), 10)
        img = cv2.addWeighted(
            img,
            1.0,
            glow,
            0.15,
            0
        )

        #kasih tone warm
        warm = np.full_like(img, (10, 20, 35))
        img = cv2.addWeighted(
            img,
            1.0,
            warm,
            0.12,
            0
        )

        #kasih float gif
        float_offset1 = int(
            10 * np.sin(frame_count * 0.1)
        )

        x1 = 420
        y1 = 120 + float_offset1

        x2 = x1 + 180
        y2 = y1 + 180

        gif1_frame = gif1_frames[gif1_index]

        img[y1:y2, x1:x2] = gif1_frame

        gif1_index += 1

        if gif1_index >= len(gif1_frames):
            gif1_index = 0

        #floating gif 2
        float_offset2 = int(
            10 * np.sin(frame_count * 0.12 + 2)
        )

        x3 = 40
        y3 = 150 + float_offset2

        x4 = x3 + 180
        y4 = y3 + 180

        gif2_frame = gif2_frames[gif2_index]

        img[y3:y4, x3:x4] = gif2_frame

        gif2_index += 1

        if gif2_index >= len(gif2_frames):
            gif2_index = 0

        #kasih emot
        particles = [
            (120, 120),
            (220, 180),
            (320, 100),
            (540, 120),
            (180, 340),
            (520, 300),
        ]

        for i, (x, y) in enumerate(particles):

            offset = int(
                6 * np.sin(frame_count * 0.08 + i)
            )

            y += offset

            cv2.circle(
                img,
                (x, y),
                7,
                (255, 255, 255),
                -1
            )

            cv2.circle(
                img,
                (x, y),
                3,
                (180, 220, 255),
                -1
            )

        #kasih emot love

        if forehead_x is not None:

            love_offset = int(
                8 * np.sin(frame_count * 0.12)
            )

            heart_y = forehead_y - 45 + love_offset

            
            cv2.circle(
                img,
                (forehead_x - 10, heart_y),
                10,
                (180, 105, 255),
                -1
            )

            cv2.circle(
                img,
                (forehead_x + 10, heart_y),
                10,
                (180, 105, 255),
                -1
            )

            points = np.array([
                [forehead_x - 20, heart_y + 3],
                [forehead_x + 20, heart_y + 3],
                [forehead_x, heart_y + 30]
            ])

            cv2.fillPoly(
                img,
                [points],
                (180, 105, 255)
            )

#open camera  
    cv2.imshow("Kicau Mania Detector",img)
    
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

#clean musiknya
pygame.mixer.music.stop()

cap.release()

cv2.destroyAllWindows()
