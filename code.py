import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")

VIDEO_PATH = "my_video.MOV"
RAW_OUTPUT = "pose_skeleton_raw.mp4"
MAX_PEOPLE = 1
CONF_THRESHOLD = 0.3

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open '{VIDEO_PATH}'. Check the name in the Files panel.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

line_thickness = max(3, width // 250)
dot_radius = max(6, width // 120)

out = cv2.VideoWriter(RAW_OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

lines_drawn = 0
dots_drawn = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, conf=0.25, verbose=False)[0]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if result.keypoints is not None and len(result.keypoints) > 0:
        points = result.keypoints.xy.cpu().numpy()
        kp_conf = result.keypoints.conf
        kp_conf = kp_conf.cpu().numpy() if kp_conf is not None else None

        box_conf = result.boxes.conf.cpu().numpy()
        best = np.argsort(-box_conf)[:MAX_PEOPLE]

        for person_idx in best:
            person = points[person_idx]

            visible = []
            for kp_idx, (x, y) in enumerate(person):
                ok = not (x == 0 and y == 0)
                if ok and kp_conf is not None and kp_conf[person_idx][kp_idx] < CONF_THRESHOLD:
                    ok = False
                visible.append(ok)

            for a, b in SKELETON:
                if visible[a] and visible[b]:
                    pt_a = (int(person[a][0]), int(person[a][1]))
                    pt_b = (int(person[b][0]), int(person[b][1]))
                    cv2.line(canvas, pt_a, pt_b, (255, 255, 255), line_thickness, cv2.LINE_AA)
                    lines_drawn += 1

            for kp_idx, (x, y) in enumerate(person):
                if visible[kp_idx]:
                    cv2.circle(canvas, (int(x), int(y)), dot_radius, (255, 255, 255), -1, cv2.LINE_AA)
                    dots_drawn += 1

    out.write(canvas)

cap.release()
out.release()
print("Done.")
print("Lines drawn:", lines_drawn, "| Dots drawn:", dots_drawn)
