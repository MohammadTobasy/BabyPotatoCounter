"""
================================================================================
 Techno Seeds — Conveyor Potato Counting via ZONE-ENTRY detection
================================================================================

APPROACH (no persistent tracking IDs across the whole video)
--------------------------------------------------------------
Instead of a single-pixel line + long-lived track IDs (fragile: an ID has to
survive dozens of frames of glare/clustering without switching), this uses a
rectangular ZONE and only ONE-FRAME lookback matching:

  - Each frame, YOLO detects potatoes (detection only, no model.track()).
  - We look at which detections fall inside the zone right now.
  - We match them against "what was inside the zone last frame" using
    simple nearest-centroid matching with a distance gate.
  - A zone-detection with NO match in the previous frame = a potato that
    just entered the zone -> COUNT IT ONCE.
  - A zone-detection that DOES match a previous-frame zone-detection is the
    same potato still sitting in the zone -> already counted, skip.
  - If a previously-in-zone object isn't seen for a couple of frames, it's
    forgotten entirely (no long-term memory, no global unique ID needed).

Why this is more robust than a thin line:
  - The zone has width, so a fast potato can't "jump over" a 1px line
    between two frames and get missed.
  - Matching only has to survive ONE frame step, not dozens -> much smaller
    failure surface than a tracker that must keep an ID alive across the
    entire time an object is on screen.

What this does NOT fix (still upstream, detection-side issues):
  - If two potatoes touch/overlap and YOLO's NMS merges them into a single
    box, that's a detection problem, not something zone/tracking logic can
    recover. Use diagnostic_detect.py first to confirm detection is stable
    before trusting any counting number from this script.

--------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------
pip install ultralytics opencv-python scipy numpy

--------------------------------------------------------------------------
CALIBRATION (do this before trusting any number)
--------------------------------------------------------------------------
1. Run with --preview on a short clip first.
2. Watch the drawn zone rectangle: it should sit somewhere on the belt
   where potatoes are LEAST clustered/overlapping, and should be wide
   enough that a potato spends at least 2-3 frames inside it (so the
   one-frame matching has a chance to work). Too thin = same failure mode
   as a line. Too wide = potatoes might enter/exit/re-enter oddly.
3. Tune, in order of importance:
     - ZONE_*_RATIO       -> position/size of the rectangle
     - MAX_MATCH_DISTANCE -> should be roughly the max px a potato moves
                              in ONE frame (belt_speed_px_per_sec / fps),
                              times ~1.5 margin
     - CONF_THRESH         -> lower if potatoes are missed entirely, raise
                              if flicker/false boxes appear
     - MAX_FRAMES_UNSEEN   -> how many frames an object can vanish
                              (occlusion/glare) inside the zone before
                              being forgotten
================================================================================
"""

import os
import time
import argparse
from collections import deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


# ============================================================================
# CONFIG — tune these
# ============================================================================
MODEL_PATH = "best.pt"
VIDEO_PATH = os.path.join("data", "DataSetS25U", "8.mp4")
OUTPUT_PATH = "output_counted_8.mp4"

CONF_THRESH = 0.30
IOU_THRESH = 0.45
IMG_SIZE = 960

# Zone as a fraction of frame width/height: (x1, y1, x2, y2), each 0..1
# Default: a horizontal band across the middle of the frame, full width.
# Adjust to match belt orientation / least-cluttered stretch.
ZONE_X1_RATIO = 0.0
ZONE_Y1_RATIO = 0.45
ZONE_X2_RATIO = 1.0
ZONE_Y2_RATIO = 0.60

MAX_MATCH_DISTANCE = 80     # px — gate for one-frame nearest-centroid matching
MAX_FRAMES_UNSEEN = 3       # frames an in-zone object can go undetected before forgotten

DEVICE = 0
HALF_PRECISION = True

DEBUG_PREVIEW = False
PRINT_EVERY_N_FRAMES = 30


# ============================================================================
# A "zone object" is intentionally lightweight — no Kalman filter, no
# long-lived global ID meant to survive the whole video. It only needs to
# persist for as long as it's physically inside the zone.
# ============================================================================
class ZoneObject:
    _next_local_id = 1

    def __init__(self, centroid, box, frame_idx):
        self.local_id = ZoneObject._next_local_id
        ZoneObject._next_local_id += 1
        self.centroid = centroid
        self.box = box
        self.last_seen_frame = frame_idx
        self.frames_unseen = 0

    def update(self, centroid, box, frame_idx):
        self.centroid = centroid
        self.box = box
        self.last_seen_frame = frame_idx
        self.frames_unseen = 0

    def mark_unseen(self):
        self.frames_unseen += 1


class ZoneCounter:
    """
    Maintains the set of objects currently believed to be inside the zone,
    matches new-frame zone-detections against them with a single-frame
    nearest-centroid gate, and counts a new object exactly once when it
    first appears in the zone with no match to the previous state.
    """

    def __init__(self, zone_xyxy):
        self.zone = zone_xyxy  # (x1, y1, x2, y2) in pixels
        self.active_objects = []  # ZoneObject instances currently tracked-in-zone
        self.total_count = 0

    @staticmethod
    def _centroid_of(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _in_zone(self, centroid):
        zx1, zy1, zx2, zy2 = self.zone
        cx, cy = centroid
        return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

    def update(self, detections, frame_idx):
        """
        detections: list of (box_xyxy, conf) for the WHOLE frame.
        Only detections whose centroid falls inside the zone are considered.
        Returns (objects_in_zone_for_drawing, newly_counted_this_frame)
        """
        zone_dets = []
        for box, conf in detections:
            c = self._centroid_of(box)
            if self._in_zone(c):
                zone_dets.append((c, box, conf))

        # ---- one-frame nearest-centroid matching against active_objects ----
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
                obj.update(c, box, frame_idx)
                matched_obj_idx.add(r)
                matched_det_idx.add(cidx)

        # ---- unmatched active objects: mark unseen, possibly drop ----
        still_active = []
        for i, obj in enumerate(self.active_objects):
            if i not in matched_obj_idx:
                obj.mark_unseen()
            if obj.frames_unseen <= MAX_FRAMES_UNSEEN:
                still_active.append(obj)
        self.active_objects = still_active

        # ---- unmatched zone detections: brand-new entries -> COUNT ----
        newly_counted = 0
        for j, (c, box, conf) in enumerate(zone_dets):
            if j not in matched_det_idx:
                new_obj = ZoneObject(c, box, frame_idx)
                self.active_objects.append(new_obj)
                self.total_count += 1
                newly_counted += 1

        return self.active_objects, newly_counted


# ============================================================================
# Drawing
# ============================================================================
def draw_frame(frame, zone_xyxy, active_objects, total_count, frame_idx, fps):
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
        cx, cy = int(obj.centroid[0]), int(obj.centroid[1])
        cv2.circle(frame, (cx, cy), 3, color, -1)

    cv2.rectangle(frame, (0, 0), (300, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"COUNT: {total_count}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(frame, f"frame {frame_idx}  fps~{fps:.1f}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


# ============================================================================
# Main
# ============================================================================
def run(model_path, video_path, output_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    zone_xyxy = (
        w * ZONE_X1_RATIO, h * ZONE_Y1_RATIO,
        w * ZONE_X2_RATIO, h * ZONE_Y2_RATIO,
    )
    print(f"[INFO] Video: {w}x{h} @ {src_fps:.2f}fps, {total_frames} frames")
    print(f"[INFO] Zone (px): {tuple(round(v, 1) for v in zone_xyxy)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))

    counter = ZoneCounter(zone_xyxy)

    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.predict(
            source=frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=IMG_SIZE,
            device=DEVICE,
            half=HALF_PRECISION,
            verbose=False,
        )

        detections = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                detections.append((tuple(box.tolist()), float(conf)))

        active_objects, newly_counted = counter.update(detections, frame_idx)

        elapsed = time.time() - t_start
        live_fps = frame_idx / elapsed if elapsed > 0 else 0.0
        frame = draw_frame(frame, zone_xyxy, active_objects, counter.total_count,
                            frame_idx, live_fps)

        writer.write(frame)

        if DEBUG_PREVIEW:
            cv2.imshow("Conveyor Potato Counter (Zone-Entry)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % PRINT_EVERY_N_FRAMES == 0 or frame_idx == total_frames:
            print(f"[{frame_idx}/{total_frames}] "
                  f"count={counter.total_count}  in_zone_now={len(active_objects)}  "
                  f"fps={live_fps:.1f}")

    cap.release()
    writer.release()
    if DEBUG_PREVIEW:
        cv2.destroyAllWindows()

    print("=" * 60)
    print(f"[DONE] Processed {frame_idx} frames in {time.time() - t_start:.1f}s")
    print(f"[RESULT] TOTAL POTATO COUNT: {counter.total_count}")
    print(f"[OUTPUT] Saved video with burn-in counter to: {output_path}")
    print("=" * 60)

    return counter.total_count


def parse_args():
    p = argparse.ArgumentParser(description="Zone-entry conveyor potato counter (YOLO11)")
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--video", default=VIDEO_PATH)
    p.add_argument("--out", default=OUTPUT_PATH)
    p.add_argument("--zone", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                   default=None,
                   help="Zone as ratios 0..1: X1 Y1 X2 Y2 (default uses config constants)")
    p.add_argument("--conf", type=float, default=CONF_THRESH)
    p.add_argument("--max-match-dist", type=float, default=MAX_MATCH_DISTANCE,
                   help="Max pixel distance for one-frame nearest-centroid matching")
    p.add_argument("--preview", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    # Windows-safe entry point
    args = parse_args()

    CONF_THRESH = args.conf
    MAX_MATCH_DISTANCE = args.max_match_dist
    DEBUG_PREVIEW = args.preview or DEBUG_PREVIEW

    if args.zone is not None:
        ZONE_X1_RATIO, ZONE_Y1_RATIO, ZONE_X2_RATIO, ZONE_Y2_RATIO = args.zone

    run(args.model, args.video, args.out)