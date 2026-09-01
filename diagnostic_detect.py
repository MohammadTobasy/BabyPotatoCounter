#!/usr/bin/env python3
"""
diagnostic_detect.py

Runs YOLO detection ONLY (no tracking).

Outputs:
    diagnostics/
        detections.csv
        detections_plot.png
        diagnostic_video.mp4
        samples/
            drop_0001.png
            ...

Author: ChatGPT
"""

import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "best.pt"
VIDEO_PATH = os.path.join("data", "DataSetS25U", "8.mp4")

OUTPUT_DIR = "diagnostics"
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")

CONF = 0.25
IOU = 0.45
IMGSZ = 960
DEVICE = 0
HALF = True

SAVE_VIDEO = True
SAVE_SAMPLE_FRAMES = True
TOP_DROPS = 20

# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

print("Loading model...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=" * 60)
print("VIDEO INFORMATION")
print("=" * 60)
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.2f}")
print(f"Frames     : {frame_count}")
print("=" * 60)

if SAVE_VIDEO:
    writer = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, "diagnostic_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

rows = []

frame_idx = 0

total_conf = 0.0
total_boxes = 0
zero_frames = 0

detections_per_frame = []

saved_frames = {}

print("\nRunning detection...\n")

while True:

    ok, frame = cap.read()

    if not ok:
        break

    frame_idx += 1

    result = model.predict(
        source=frame,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=DEVICE,
        half=HALF,
        verbose=False,
    )[0]

    det_count = 0
    confs = []

    if result.boxes is not None and len(result.boxes):

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        det_count = len(boxes)

        for box, score in zip(boxes, scores):

            x1, y1, x2, y2 = map(int, box)

            confs.append(float(score))

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                1,
            )

    if det_count == 0:
        zero_frames += 1

    detections_per_frame.append(det_count)

    avg_conf = np.mean(confs) if len(confs) else 0
    min_conf = np.min(confs) if len(confs) else 0
    max_conf = np.max(confs) if len(confs) else 0

    total_boxes += det_count
    total_conf += sum(confs)

    rows.append([
        frame_idx,
        det_count,
        avg_conf,
        min_conf,
        max_conf,
    ])

    saved_frames[frame_idx] = frame.copy()

    cv2.putText(
        frame,
        f"Frame {frame_idx}",
        (20,35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2,
    )

    cv2.putText(
        frame,
        f"Detections: {det_count}",
        (20,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2,
    )

    if SAVE_VIDEO:
        writer.write(frame)

    if frame_idx % 100 == 0:
        print(f"{frame_idx}/{frame_count}")

cap.release()

if SAVE_VIDEO:
    writer.release()

csv_path = os.path.join(OUTPUT_DIR, "detections.csv")

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "frame",
        "detections",
        "avg_conf",
        "min_conf",
        "max_conf",
    ])

    writer.writerows(rows)

drops = []

for i in range(1, len(detections_per_frame)):

    diff = detections_per_frame[i-1] - detections_per_frame[i]

    drops.append((diff, i+1))

drops.sort(reverse=True)

print("\nLargest Drops")

for diff, frame in drops[:TOP_DROPS]:

    print(
        f"Frame {frame} drop={diff}"
    )

    if SAVE_SAMPLE_FRAMES:

        cv2.imwrite(
            os.path.join(
                SAMPLES_DIR,
                f"drop_{frame:06d}.png"
            ),
            saved_frames[frame],
        )

plt.figure(figsize=(14,6))

plt.plot(detections_per_frame)

plt.title("Detections per Frame")

plt.xlabel("Frame")

plt.ylabel("Detections")

plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "detections_plot.png"
    ),
    dpi=300,
)

plt.close()

print("\n" + "="*60)

print("SUMMARY")

print("="*60)

print(f"Frames processed : {frame_count}")

print(f"Total detections : {total_boxes}")

print(f"Average/frame    : {np.mean(detections_per_frame):.2f}")

print(f"Maximum/frame    : {np.max(detections_per_frame)}")

print(f"Minimum/frame    : {np.min(detections_per_frame)}")

print(f"Average conf     : {total_conf/max(total_boxes,1):.3f}")

print(f"Zero frames      : {zero_frames}")

print(f"Zero %           : {100*zero_frames/frame_count:.2f}%")

print("="*60)

print("\nSaved to:")
print(OUTPUT_DIR)