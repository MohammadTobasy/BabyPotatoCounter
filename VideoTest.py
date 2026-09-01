"""
================================================================================
 Techno Seeds — Recorded Video Test Runner (offline, NOT real-time)
 Runs the exact same detection + zone-entry counting logic used in the live
 counter, but reads frames from a video FILE on disk instead of a live
 camera, and processes them as fast as the hardware allows (no real-time
 pacing, no camera reconnect logic, no Start/Pause/Review workflow — this
 is a one-shot accuracy test, not the production counter).

 USE CASE:
   You have a recorded clip (e.g. a phone video) and want to see how the
   model + counting logic performs on it, without touching the live camera
   pipeline at all. Good for quickly comparing "webcam live" vs "phone
   video" accuracy side by side using the SAME model and SAME counting
   code, which is exactly the comparison you're trying to make.

 USAGE:
   python process_video.py --video my_test_clip.mp4
   python process_video.py --video my_test_clip.mp4 --model best26.pt --conf 0.44
   python process_video.py --video my_test_clip.mp4 --no-preview --output annotated.avi

 It reuses your saved zone_config.json (same one --calibrate produces for
 the live script) if present, so the counting zone lines up the same way.
 If the recorded video has a different resolution/aspect ratio than your
 webcam, the zone ratios still apply (they're relative 0-1 fractions of
 frame size), but re-check the zone placement visually via --no-preview
 removed (i.e. WITH preview) the first time on a new video source.

 At the end it prints: total counted, total frames processed, and the
 processing throughput (frames/sec on this hardware) — note this last
 number is NOT the live camera FPS, it's just how fast this machine can
 chew through the file, useful for sanity-checking hardware speed but not
 a live-performance number.
================================================================================
"""

import os
import sys
import json
import time
import argparse

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ============================================================================
# CONFIG (same meaning as the live scripts — kept here so this file is
# fully standalone and doesn't depend on importing the other script)
# ============================================================================
ZONE_CONFIG_PATH = "zone_config.json"

ZONE_X1_RATIO = 0.0
ZONE_Y1_RATIO = 0.45
ZONE_X2_RATIO = 1.0
ZONE_Y2_RATIO = 0.60

TARGET_CLASS_NAME = "Baby Potato"

MAX_MATCH_DISTANCE = 80
MAX_FRAMES_UNSEEN = 6
MIN_BOX_AREA_PX = 400
MAX_BOX_AREA_PX = 60000
STRICT_CONF_THRESH = 0.45
WARMUP_FRAMES = 15
GLARE_MIN_VALUE = 210
GLARE_MAX_SATURATION = 40


def load_zone_ratios():
    global ZONE_X1_RATIO, ZONE_Y1_RATIO, ZONE_X2_RATIO, ZONE_Y2_RATIO
    if os.path.exists(ZONE_CONFIG_PATH):
        with open(ZONE_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        ZONE_X1_RATIO = cfg["x1"]
        ZONE_Y1_RATIO = cfg["y1"]
        ZONE_X2_RATIO = cfg["x2"]
        ZONE_Y2_RATIO = cfg["y2"]
        print(f"[INFO] Loaded zone from {ZONE_CONFIG_PATH}: "
              f"{ZONE_X1_RATIO:.3f},{ZONE_Y1_RATIO:.3f},{ZONE_X2_RATIO:.3f},{ZONE_Y2_RATIO:.3f}")
    else:
        print("[INFO] No zone_config.json found — using default zone ratios "
              "(top-ish horizontal band across the frame). Check visually with "
              "the preview window that this lines up on your test video.")


# ============================================================================
# Zone counting (identical logic to the live scripts)
# ============================================================================
class ZoneObject:
    _next_local_id = 1

    def __init__(self, centroid, box, frame_idx, conf=None):
        self.local_id = ZoneObject._next_local_id
        ZoneObject._next_local_id += 1
        self.centroid = centroid
        self.box = box
        self.frames_unseen = 0
        self.last_conf = conf

    def update(self, centroid, box, conf=None):
        self.centroid = centroid
        self.box = box
        self.frames_unseen = 0
        if conf is not None:
            self.last_conf = conf

    def mark_unseen(self):
        self.frames_unseen += 1


class ZoneCounter:
    def __init__(self, zone_xyxy):
        self.zone = zone_xyxy
        self.active_objects = []
        self.total_count = 0

    @staticmethod
    def _centroid_of(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _is_glare(frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False
        region = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))
        return mean_v >= GLARE_MIN_VALUE and mean_s <= GLARE_MAX_SATURATION

    def _in_zone(self, centroid):
        zx1, zy1, zx2, zy2 = self.zone
        cx, cy = centroid
        return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

    def update(self, detections, frame_idx, frame, debug=False):
        zone_dets = []
        for box, conf in detections:
            x1, y1, x2, y2 = box
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if area < MIN_BOX_AREA_PX or area > MAX_BOX_AREA_PX:
                continue
            if self._is_glare(frame, box):
                continue
            c = self._centroid_of(box)
            if self._in_zone(c):
                zone_dets.append((c, box, conf))

        matched_obj_idx = set()
        matched_det_idx = set()

        if self.active_objects and zone_dets:
            cost = np.zeros((len(self.active_objects), len(zone_dets)), dtype=np.float32)
            for i, obj in enumerate(self.active_objects):
                ox, oy = obj.centroid
                for j, (c, box, conf) in enumerate(zone_dets):
                    dx, dy = c
                    dist = np.hypot(ox - dx, oy - dy)
                    cost[i, j] = dist if dist <= MAX_MATCH_DISTANCE else 1e6

            row_idx, col_idx = linear_sum_assignment(cost)
            for r, cidx in zip(row_idx, col_idx):
                if cost[r, cidx] >= 1e6:
                    continue
                obj = self.active_objects[r]
                c, box, conf = zone_dets[cidx]
                obj.update(c, box, conf)
                matched_obj_idx.add(r)
                matched_det_idx.add(cidx)

        still_active = []
        for i, obj in enumerate(self.active_objects):
            if i not in matched_obj_idx:
                obj.mark_unseen()
            if obj.frames_unseen <= MAX_FRAMES_UNSEEN:
                still_active.append(obj)
        self.active_objects = still_active

        for j, (c, box, conf) in enumerate(zone_dets):
            if j not in matched_det_idx:
                if conf < STRICT_CONF_THRESH:
                    continue
                new_obj = ZoneObject(c, box, frame_idx, conf)
                self.active_objects.append(new_obj)
                if frame_idx > WARMUP_FRAMES:
                    self.total_count += 1
                    if debug:
                        print(f"[DEBUG] Counted #{self.total_count} at frame {frame_idx} conf={conf:.3f}")

        return self.active_objects


# ============================================================================
# Drawing (simplified — no bag/review UI, just zone + boxes + running count)
# ============================================================================
def draw_frame(frame, zone_xyxy, active_objects, total_count, proc_fps, frame_idx, total_frames):
    h, w = frame.shape[:2]
    zx1, zy1, zx2, zy2 = map(int, zone_xyxy)
    overlay = frame.copy()
    cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (255, 0, 0), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)

    for obj in active_objects:
        x1, y1, x2, y2 = map(int, obj.box)
        color = (0, 255, 0) if obj.frames_unseen == 0 else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.rectangle(frame, (0, 0), (320, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"COUNT: {total_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame {frame_idx}/{total_frames}  proc-fps:{proc_fps:.1f}",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame


# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Offline test runner: process a recorded video file, not a live camera")
    p.add_argument("--video", type=str, required=True, help="Path to the recorded video file to test")
    p.add_argument("--model", type=str, default="best26.pt", help="Model checkpoint path (.pt, or an OpenVINO export folder)")
    p.add_argument("--conf", type=float, default=0.44)
    p.add_argument("--iou", type=float, default=0.55)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None,
                    help="'cuda:0' or 'cpu'. Default: auto-detect (uses GPU if available).")
    p.add_argument("--no-half", action="store_true", help="Disable FP16 (only relevant on GPU)")
    p.add_argument("--no-preview", action="store_true", help="Don't show a live preview window while processing (faster)")
    p.add_argument("--output", type=str, default="processed_video_output.avi",
                    help="Path to save the annotated output video")
    p.add_argument("--no-save", action="store_true", help="Don't save an annotated output video")
    p.add_argument("--debug", action="store_true", help="Log confidence of each counted potato")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    # Auto-detect device if not specified
    device = args.device
    if device is None:
        device = "cuda:0" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
    half = (not args.no_half) and device.startswith("cuda")

    print(f"[INFO] Loading model: {args.model} on device={device} half={half}")
    model = YOLO(args.model)
    if device.startswith("cuda") and _HAS_TORCH:
        model.to(device)

    load_zone_ratios()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {w}x{h} @ {src_fps:.1f}fps, {total_frames} frames "
          f"(~{total_frames / src_fps:.1f}s)")

    zone_xyxy = (
        w * ZONE_X1_RATIO, h * ZONE_Y1_RATIO,
        w * ZONE_X2_RATIO, h * ZONE_Y2_RATIO,
    )
    print(f"[INFO] Zone (px): {tuple(round(v, 1) for v in zone_xyxy)}")

    counter = ZoneCounter(zone_xyxy)

    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (w, h))
        if not writer.isOpened():
            print(f"[WARN] Could not open writer for {args.output} — continuing without saving.")
            writer = None

    if not args.no_preview:
        cv2.namedWindow("Video Test — press q to stop early", cv2.WINDOW_NORMAL)

    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of video
        frame_idx += 1

        results = model.predict(
            source=frame, conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=device, half=half, verbose=False,
        )
        detections = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()
            for box, conf, cls_id in zip(xyxy, confs, cls_ids):
                class_name = model.names[int(cls_id)]
                if class_name != TARGET_CLASS_NAME:
                    continue
                detections.append((tuple(box.tolist()), float(conf)))

        active_objects = counter.update(detections, frame_idx, frame, debug=args.debug)

        elapsed = time.time() - t_start
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0

        annotated = draw_frame(frame.copy(), zone_xyxy, active_objects, counter.total_count,
                                proc_fps, frame_idx, total_frames)

        if writer is not None:
            writer.write(annotated)
        if not args.no_preview:
            cv2.imshow("Video Test — press q to stop early", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Stopped early by user.")
                break

        if frame_idx % 30 == 0:
            print(f"[{frame_idx}/{total_frames}] count={counter.total_count}  proc_fps={proc_fps:.1f}")

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_preview:
        cv2.destroyAllWindows()

    total_elapsed = time.time() - t_start
    print("=" * 60)
    print(f"[RESULT] TOTAL POTATOES COUNTED: {counter.total_count}")
    print(f"[RESULT] Frames processed: {frame_idx} in {total_elapsed:.1f}s "
          f"(avg processing throughput: {frame_idx / total_elapsed:.1f} fps on this hardware)")
    if writer is not None:
        print(f"[OUTPUT] Annotated video saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()