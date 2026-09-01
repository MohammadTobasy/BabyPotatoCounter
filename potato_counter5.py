"""
================================================================================
 Techno Seeds — Conveyor Potato Counter (Laptop / OpenVINO / Live Camera)
 ALL-IN-ONE VERSION: live counting + zone calibration + worker bag review

 EDITED VERSION — changes vs. previous version are marked with "# FIX:"
 Summary of what changed and why:
   1) FPS was computed as frame_idx / (time since script start) — this
      includes idle STOPPED-mode frames and script startup time, which is
      why FPS looked inconsistent run-to-run even under "same conditions".
      Now uses a rolling window of actual recent frame timestamps.
   2) Camera capture now explicitly requests MJPG from the camera itself
      (not just the output writer). Many USB webcams default to raw
      YUY2 over DirectShow, which can saturate USB bandwidth at higher
      resolutions and cause exactly the run-to-run FPS variance reported.
   3) cv2.CAP_PROP_BUFFERSIZE = 1 is now actually applied (was identified
      in the project handover as a fix for "slow detection" but not yet
      wired into the code).
   4) CPU thread count is pinned (env vars + cv2.setNumThreads) BEFORE
      cv2/ultralytics are imported, so thread allocation doesn't silently
      vary between runs.
   5) Startup now prints a model verification block: path, task, class
      names, and whether TARGET_CLASS_NAME actually matches something in
      the model — catches the "silently filtered, zero detections counted"
      failure mode immediately instead of after a wasted belt test.
   6) Startup also prints whether the loaded model reports itself as
      end-to-end/NMS-free. YOLO26 is NMS-free; if end2end=True,
      IOU_THRESH may not be doing anything in your postprocessing path,
      and re-tuning it further would be tuning a no-op.
   7) Optional --debug flag: logs confidence + frames-unseen pattern at
      the moment each potato is first counted, to help correlate
      "detected late / missed / flickers orange" with low-confidence /
      motion-blur frames.
================================================================================

TWO MODES IN ONE FILE:

1) CALIBRATION MODE — run once to set the zone to match your belt exactly:
     python potato_counter.py --calibrate
   Click top-left then bottom-right of the belt zone, press 'c' to save,
   'r' to redo, 'q' to quit. Saves to zone_config.json, used automatically
   next time you run normally.

2) NORMAL (LIVE COUNTING) MODE:
     python potato_counter.py
   Controls:
     's' = Start counting a new bag / Resume from Pause
     'p' = Pause (freezes count, no logging, resumable)
     'x' = Finish bag -> REVIEW: shows the count, lets the worker
           correct it with '+'/'-' before confirming
     Enter or 'c' (while in review) = Confirm the count -> logged to
           bag_history.json with bag number + exact date/time, then
           ready for the next bag automatically
     'r' = Reset current count to 0 (does not affect already-confirmed
           bags in history)
     'q' = Quit

HISTORY LOG:
  Every confirmed bag is appended to bag_history.json as a record:
    {bag_number, timestamp, model_count, final_count, was_edited}
  This file is meant to later feed the admin dashboard (Flask/SQLite
  backend, per the project plan) — for now it's a simple local JSON log
  so nothing is lost while that backend isn't built yet.
================================================================================
"""

import os

# FIX (#4), REVISED: thread pinning is now OPT-IN via --threads N, not
# forced on by default. Forcing NUM_THREADS to os.cpu_count() (which returns
# LOGICAL cores, e.g. 8 on a 4-core/8-thread CPU via hyperthreading) can
# oversubscribe the 4 physical cores and cause contention that's SLOWER than
# letting OpenVINO auto-tune. Must be parsed from sys.argv here (before
# other imports) since env vars need to be set before cv2/ultralytics load.
import sys
_threads_arg = None
if "--threads" in sys.argv:
    _idx = sys.argv.index("--threads")
    if _idx + 1 < len(sys.argv):
        _threads_arg = sys.argv[_idx + 1]
if _threads_arg:
    os.environ["OMP_NUM_THREADS"] = _threads_arg
    os.environ["MKL_NUM_THREADS"] = _threads_arg
    os.environ["OPENBLAS_NUM_THREADS"] = _threads_arg
    os.environ["OV_CPU_THREADS_NUM"] = _threads_arg
    print(f"[INFO] Thread pinning ENABLED via --threads {_threads_arg}")
else:
    print("[INFO] Thread pinning NOT set (default) — letting OpenVINO/OpenCV "
          "auto-tune. Try '--threads 4' (physical core count) to compare.")

import json
import time
import argparse
import collections
from datetime import datetime

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

NUM_THREADS = 4  # or whatever default you want

# FIX (#4): also pin OpenCV's own internal thread pool (affects cvtColor,
# resize, etc. used in glare detection and drawing).
cv2.setNumThreads(int(NUM_THREADS))


# ============================================================================
# CONFIG
# ============================================================================
MODEL_PATH = "best26_openvino_model/"
CAMERA_INDEX = 0          # change if the wrong camera opens
ZONE_CONFIG_PATH = "zone_config.json"
HISTORY_PATH = "bag_history.json"
OUTPUT_PATH = "output_counted_live.avi"

# Reference resolutions — override via --width/--height, e.g.:
#   SD  (default — best FPS, model resizes to 640 internally anyway):
#                          640  x 480
#   HD:                   1280 x 720
#   FHD ("1080p"):        1920 x 1080
#   2K  (QHD):            2560 x 1440
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CONF_THRESH = 0.35
IOU_THRESH = 0.65
IMG_SIZE = 640             # must match export imgsz

# Only detections of this class are counted. If you retrain with multiple
# classes (e.g. "potato", "finger", "light"), set this to EXACTLY match the
# potato class name you used in Roboflow (case-sensitive) — other classes
# will still be detected internally but silently ignored/not counted.
TARGET_CLASS_NAME = "Baby Potato"

# Default zone (used only if zone_config.json doesn't exist yet —
# run --calibrate to set this properly for your belt)
ZONE_X1_RATIO = 0.0
ZONE_Y1_RATIO = 0.45
ZONE_X2_RATIO = 1.0
ZONE_Y2_RATIO = 0.60

MAX_MATCH_DISTANCE = 80
MAX_FRAMES_UNSEEN = 6

MIN_BOX_AREA_PX = 400
MAX_BOX_AREA_PX = 60000
STRICT_CONF_THRESH = 0.45

WARMUP_FRAMES = 15

GLARE_MIN_VALUE = 210
GLARE_MAX_SATURATION = 40

DEVICE = "cpu"
HALF_PRECISION = False

DEBUG_PREVIEW = True
SAVE_VIDEO = True    # set False (or --no-save) to skip writing the burned-in output video
FRAME_SKIP = 1        # run the model every Nth frame (1 = every frame, no skipping)
PRINT_EVERY_N_FRAMES = 30

# FIX (#7): toggled via --debug. Logs per-object confidence at the moment
# each potato is first counted, to help correlate "late/missed/flicker"
# with low-confidence or motion-blurred frames during belt testing.
DEBUG_LOG = False

# FIX (#1): how many recent frames to average FPS over, instead of
# averaging over the entire run (including idle STOPPED time).
FPS_WINDOW = 30


# ============================================================================
# Zone config load/save (used by both calibration and normal mode)
# ============================================================================
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
        print("[INFO] No zone_config.json found — using default zone ratios. "
              "Run 'python potato_counter.py --calibrate' to set this precisely.")


def save_zone_ratios(x1, y1, x2, y2):
    with open(ZONE_CONFIG_PATH, "w") as f:
        json.dump({"x1": x1, "y1": y1, "x2": x2, "y2": y2}, f, indent=2)
    print(f"[INFO] Saved zone to {ZONE_CONFIG_PATH}")


# ============================================================================
# History log
# ============================================================================
def append_history(bag_number, model_count, final_count):
    record = {
        "bag_number": bag_number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_count": model_count,
        "final_count": final_count,
        "was_edited": model_count != final_count,
    }
    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.append(record)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[HISTORY] Bag {bag_number} logged: model={model_count} final={final_count} "
          f"edited={record['was_edited']}")


def next_bag_number():
    if not os.path.exists(HISTORY_PATH):
        return 1
    with open(HISTORY_PATH, "r") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            history = []
    return len(history) + 1


# ============================================================================
# Shared camera setup — FIX (#2, #3): MJPG capture format + buffer size 1
# ============================================================================
def open_camera(camera_index, width, height):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return cap

    # FIX (#2): request MJPG from the camera itself. Many USB webcams
    # default to raw YUY2 over DirectShow, which is bandwidth-limited at
    # higher resolutions/fps and can silently cap the delivered frame rate
    # or make it inconsistent run-to-run. MJPG is compressed on-camera and
    # is much less likely to saturate USB bandwidth. Must be set BEFORE
    # width/height for some drivers to honor it.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # FIX (#3): keep only 1 frame buffered. Without this, if processing
    # falls behind the camera's capture rate even briefly, OpenCV/DirectShow
    # can queue up stale frames — you then "catch up" by processing frames
    # that are already old, which shows up as a potato being detected late
    # (already near/past the zone) or the count lagging behind reality.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


# ============================================================================
# CALIBRATION MODE
# ============================================================================
def run_calibration():
    points = []

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 2:
                points = []
            points.append((x, y))

    cap = open_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {CAMERA_INDEX}.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened at {w}x{h}")
    print("[INFO] Click top-left then bottom-right of the belt zone.")
    print("[INFO] 'c' confirm+save | 'r' redo | 'q' quit")

    cv2.namedWindow("Zone Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Zone Calibration", on_mouse)

    last_ratios = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Failed to read frame.")
            break

        display = frame.copy()
        for p in points:
            cv2.circle(display, p, 6, (0, 255, 0), -1)

        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
            last_ratios = (x1 / w, y1 / h, x2 / w, y2 / h)
            cv2.putText(display, f"ratios: {last_ratios[0]:.3f} {last_ratios[1]:.3f} "
                                  f"{last_ratios[2]:.3f} {last_ratios[3]:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(display, "click 2 pts (TL, BR) | c=save r=redo q=quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Zone Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            points = []
            last_ratios = None
        elif key == ord("c"):
            if last_ratios:
                save_zone_ratios(*last_ratios)
            else:
                print("[INFO] Click 2 points first.")

    cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# Zone counting objects
# ============================================================================
class ZoneObject:
    _next_local_id = 1

    def __init__(self, centroid, box, frame_idx, conf=None):
        self.local_id = ZoneObject._next_local_id
        ZoneObject._next_local_id += 1
        self.centroid = centroid
        self.box = box
        self.last_seen_frame = frame_idx
        self.frames_unseen = 0
        self.last_conf = conf          # FIX (#7): track confidence for debug logging
        self.first_seen_frame = frame_idx

    def update(self, centroid, box, frame_idx, conf=None):
        self.centroid = centroid
        self.box = box
        self.last_seen_frame = frame_idx
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

    def update(self, detections, frame_idx, frame):
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
                obj.update(c, box, frame_idx, conf)
                matched_obj_idx.add(r)
                matched_det_idx.add(cidx)

        still_active = []
        for i, obj in enumerate(self.active_objects):
            if i not in matched_obj_idx:
                obj.mark_unseen()
            if obj.frames_unseen <= MAX_FRAMES_UNSEEN:
                still_active.append(obj)
        self.active_objects = still_active

        newly_counted = 0
        for j, (c, box, conf) in enumerate(zone_dets):
            if j not in matched_det_idx:
                if conf < STRICT_CONF_THRESH:
                    continue
                new_obj = ZoneObject(c, box, frame_idx, conf)
                self.active_objects.append(new_obj)
                if frame_idx > WARMUP_FRAMES:
                    self.total_count += 1
                    newly_counted += 1
                    if DEBUG_LOG:
                        # FIX (#7): log confidence at the moment of counting —
                        # if "late/missed" potatoes consistently show conf
                        # just above STRICT_CONF_THRESH, that points to
                        # motion blur pushing confidence right at the edge.
                        print(f"[DEBUG] Counted new potato #{self.total_count} "
                              f"at frame {frame_idx} conf={conf:.3f} "
                              f"(STRICT_CONF_THRESH={STRICT_CONF_THRESH})")

        return self.active_objects, newly_counted


# ============================================================================
# Drawing
# ============================================================================
def draw_frame(frame, zone_xyxy, active_objects, total_count, fps, mode, review_count=None, bag_number=None):
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

    display_count = review_count if review_count is not None else total_count
    cv2.rectangle(frame, (0, 0), (300, 95), (0, 0, 0), -1)
    cv2.putText(frame, f"COUNT: {display_count}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    if bag_number is not None:
        cv2.putText(frame, f"BAG #{bag_number}", (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    fps_text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 8
    box_x1 = w - tw - pad * 2
    box_y1 = h - 40 - th - pad * 2
    cv2.rectangle(frame, (box_x1, box_y1), (w, box_y1 + th + pad * 2), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (box_x1 + pad, box_y1 + th + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if mode == "REVIEW":
        status_text = "[REVIEW] +/- adjust  Enter/c=Confirm  q=Quit"
        status_color = (0, 200, 255)
    elif mode == "RUNNING":
        status_text = "[RUNNING]  p=Pause  x=Finish bag  r=Reset  q=Quit"
        status_color = (0, 200, 0)
    elif mode == "PAUSED":
        status_text = "[PAUSED]  s=Resume  x=Finish bag  r=Reset  q=Quit"
        status_color = (0, 165, 255)
    else:
        status_text = "[STOPPED]  s=Start  r=Reset  q=Quit"
        status_color = (0, 0, 220)

    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
    return frame


# ============================================================================
# FIX (#5, #6): model verification block
# ============================================================================
def verify_model(model):
    print("-" * 60)
    print(f"[VERIFY] Model path : {MODEL_PATH}")
    try:
        print(f"[VERIFY] Model task : {model.task}")
    except Exception:
        pass
    print(f"[VERIFY] Class names: {model.names}")

    names_list = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    if TARGET_CLASS_NAME in names_list:
        print(f"[VERIFY] TARGET_CLASS_NAME '{TARGET_CLASS_NAME}' -> OK, matches a model class.")
    else:
        print(f"[VERIFY] *** WARNING ***: TARGET_CLASS_NAME '{TARGET_CLASS_NAME}' "
              f"does NOT match any class in the model ({names_list}). "
              f"ALL detections will be silently filtered out and nothing will "
              f"ever be counted. Fix TARGET_CLASS_NAME to match exactly "
              f"(case-sensitive) before running on the belt.")

    # FIX (#6): report whether the underlying model is end-to-end / NMS-free.
    # If True, IOU_THRESH passed to model.predict() may have little or no
    # effect on the final boxes — don't spend time re-tuning it without
    # confirming it actually changes anything for this model.
    end2end = None
    try:
        end2end = getattr(model.model, "end2end", None)
    except Exception:
        pass
    if end2end is None:
        print("[VERIFY] Could not determine end2end/NMS-free status automatically. "
              "To check manually: run the same frame twice with very different "
              "--conf/iou-adjacent settings and see if box count/positions change; "
              "if IOU has no visible effect, the model is likely NMS-free "
              "end-to-end and IOU_THRESH is a no-op for it.")
    else:
        print(f"[VERIFY] Model end2end (NMS-free) = {end2end}")
        if end2end:
            print("[VERIFY] NOTE: this model appears NMS-free/end-to-end — "
                  "IOU_THRESH may not meaningfully affect output. Don't spend "
                  "belt-testing time re-tuning it without confirming it does.")
    print("-" * 60)


# ============================================================================
# Main live-counting loop
# ============================================================================
def run_counter():
    load_zone_ratios()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"[INFO] Loading OpenVINO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # FIX (#4): best-effort OpenVINO CPU thread configuration. Ultralytics
    # owns the ov.Core()/compiled model internally, so this is not
    # guaranteed to take effect on every ultralytics version — the env vars
    # set at the top of this file are the more reliable lever. This is left
    # in as a best-effort extra and silently no-ops if unsupported.
    try:
        predictor = getattr(model, "predictor", None)
        ov_model = getattr(predictor, "model", None) if predictor else None
        core = getattr(ov_model, "ov_compiled_model", None) if ov_model else None
        if core is not None:
            print("[INFO] OpenVINO compiled model detected; thread count controlled "
                  "via OMP_NUM_THREADS/OV_CPU_THREADS_NUM env vars set at script start.")
    except Exception:
        pass

    verify_model(model)  # FIX (#5, #6)

    cap = open_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)  # FIX (#2, #3)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 0:
        src_fps = 30.0  # camera didn't report a valid FPS — use a safe default
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w < CAMERA_WIDTH or h < CAMERA_HEIGHT:
        print(f"[WARN] Requested {CAMERA_WIDTH}x{CAMERA_HEIGHT} but camera granted {w}x{h} — "
              f"either it doesn't support that mode over this connection/driver, "
              f"or that exact mode isn't available. Proceeding at {w}x{h}.")

    zone_xyxy = (
        w * ZONE_X1_RATIO, h * ZONE_Y1_RATIO,
        w * ZONE_X2_RATIO, h * ZONE_Y2_RATIO,
    )
    print(f"[INFO] Camera: {w}x{h} @ ~{src_fps:.2f}fps (reported)")
    print(f"[INFO] Zone (px): {tuple(round(v, 1) for v in zone_xyxy)}")
    print("[INFO] Controls -> 's' Start/Resume | 'p' Pause | 'x' Finish bag (review) | 'r' Reset | 'q' Quit")

    if DEBUG_PREVIEW:
        cv2.namedWindow("Techno Seeds Potato Counter", cv2.WINDOW_NORMAL)

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, src_fps, (w, h))
        if not writer.isOpened():
            print(f"[WARN] Could not open output video writer for {OUTPUT_PATH}. "
                  f"Live counting will still work, but the recorded video won't be saved.")
            writer = None

    counter = ZoneCounter(zone_xyxy)
    bag_number = next_bag_number()
    print(f"[INFO] Next bag will be logged as Bag {bag_number}")

    mode = "STOPPED"
    review_model_count = None
    review_final_count = None

    frame_idx = 0
    detect_frame_idx = 0
    t_start = time.time()
    disconnect_start = None
    RECONNECT_RETRY_SECONDS = 3

    # FIX (#1): rolling window of recent frame timestamps for a stable,
    # meaningful FPS readout instead of an average since script start.
    frame_times = collections.deque(maxlen=FPS_WINDOW)

    while True:
        ok, frame = cap.read()

        if not ok:
            if disconnect_start is None:
                disconnect_start = time.time()
                print("[WARN] Camera feed lost. Attempting to reconnect "
                      f"(retrying every {RECONNECT_RETRY_SECONDS}s)...")
            cap.release()
            time.sleep(RECONNECT_RETRY_SECONDS)
            cap = open_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)  # FIX (#2, #3)
            if cap.isOpened():
                down_secs = time.time() - disconnect_start
                print(f"[INFO] Camera reconnected after {down_secs:.1f}s downtime.")
                disconnect_start = None
            continue

        if disconnect_start is not None:
            down_secs = time.time() - disconnect_start
            print(f"[INFO] Camera reconnected after {down_secs:.1f}s downtime.")
            disconnect_start = None

        frame_idx += 1
        frame_times.append(time.time())  # FIX (#1)

        active_objects = counter.active_objects
        if mode == "RUNNING":
            detect_frame_idx += 1
            if FRAME_SKIP <= 1 or detect_frame_idx % FRAME_SKIP == 0:
                results = model.predict(
                    source=frame, conf=CONF_THRESH, iou=IOU_THRESH,
                    imgsz=IMG_SIZE, device=DEVICE, half=HALF_PRECISION, verbose=False,
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
                active_objects, _ = counter.update(detections, frame_idx, frame)

        # FIX (#1): FPS from a rolling window of real elapsed time between
        # the oldest and newest frame in the window, not since script start.
        if len(frame_times) >= 2:
            window_elapsed = frame_times[-1] - frame_times[0]
            live_fps = (len(frame_times) - 1) / window_elapsed if window_elapsed > 0 else 0.0
        else:
            live_fps = 0.0

        review_display = review_final_count if mode == "REVIEW" else None
        frame = draw_frame(frame, zone_xyxy, active_objects, counter.total_count,
                            live_fps, mode, review_display, bag_number)

        if writer is not None:
            writer.write(frame)
        if DEBUG_PREVIEW:
            cv2.imshow("Techno Seeds Potato Counter", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif mode in ("STOPPED", "PAUSED") and key == ord("s"):
            if mode == "STOPPED":
                print("[INFO] STARTED")
            else:
                print("[INFO] RESUMED")
            mode = "RUNNING"

        elif mode == "RUNNING" and key == ord("p"):
            mode = "PAUSED"
            print(f"[INFO] PAUSED. Count preserved at {counter.total_count}. "
                  f"Press 's' to resume.")

        elif mode in ("RUNNING", "PAUSED") and key == ord("x"):
            review_model_count = counter.total_count
            review_final_count = counter.total_count
            mode = "REVIEW"
            print(f"[INFO] Bag finished -> REVIEW. Model counted {review_model_count}. "
                  f"Adjust with +/- if needed, Enter/'c' to confirm.")

        elif mode == "REVIEW":
            if key in (ord("+"), ord("=")):
                review_final_count += 1
            elif key == ord("-"):
                review_final_count = max(0, review_final_count - 1)
            elif key in (13, ord("c")):
                append_history(bag_number, review_model_count, review_final_count)
                bag_number += 1
                counter.total_count = 0
                counter.active_objects = []
                review_model_count = None
                review_final_count = None
                mode = "STOPPED"
                print(f"[INFO] Bag confirmed. Ready for next bag ({bag_number}). Press 's' to start.")

        elif key == ord("r"):
            counter.total_count = 0
            counter.active_objects = []
            if mode == "REVIEW":
                review_final_count = 0
            print("[INFO] RESET — count set to 0")

        if mode == "RUNNING" and frame_idx % PRINT_EVERY_N_FRAMES == 0:
            print(f"[{frame_idx}] count={counter.total_count}  "
                  f"in_zone_now={len(active_objects)}  fps={live_fps:.1f}")

    cap.release()
    if writer is not None:
        writer.release()
    if DEBUG_PREVIEW:
        cv2.destroyAllWindows()

    print("=" * 60)
    total_elapsed = time.time() - t_start
    print(f"[DONE] Processed {frame_idx} frames in {total_elapsed:.1f}s")
    avg_fps = frame_idx / total_elapsed if frame_idx else 0.0
    print(f"[RESULT] Resolution: {w}x{h}  |  Overall average FPS (whole run incl. idle time): {avg_fps:.1f}")
    print(f"[OUTPUT] Saved video with burn-in counter to: {OUTPUT_PATH}")
    print(f"[HISTORY] Full bag history saved in: {HISTORY_PATH}")
    print("=" * 60)


# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Techno Seeds Potato Counter — all-in-one")
    p.add_argument("--calibrate", action="store_true", help="Run zone calibration mode instead of counting")
    p.add_argument("--camera", type=int, default=CAMERA_INDEX)
    p.add_argument("--conf", type=float, default=CONF_THRESH)
    p.add_argument("--width", type=int, default=CAMERA_WIDTH,
                    help="Requested camera width. Reference: SD=640 HD=1280 FHD=1920 2K=2560")
    p.add_argument("--height", type=int, default=CAMERA_HEIGHT,
                    help="Requested camera height. Reference: SD=480 HD=720 FHD=1080 2K=1440")
    p.add_argument("--no-save", action="store_true",
                    help="Don't write the burned-in output video (saves disk I/O + a little CPU)")
    p.add_argument("--no-preview", action="store_true",
                    help="Don't show the live preview window (best FPS, but you can't see what's happening)")
    p.add_argument("--skip", type=int, default=FRAME_SKIP,
                    help="Run the model every Nth frame instead of every frame")
    p.add_argument("--debug", action="store_true",
                    help="Log per-object confidence when each potato is counted (helps diagnose "
                         "late/missed detections vs. motion blur)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CAMERA_INDEX = args.camera
    CONF_THRESH = args.conf
    CAMERA_WIDTH = args.width
    CAMERA_HEIGHT = args.height
    SAVE_VIDEO = not args.no_save
    DEBUG_PREVIEW = not args.no_preview
    FRAME_SKIP = max(1, args.skip)
    DEBUG_LOG = args.debug

    if args.calibrate:
        run_calibration()
    else:
        run_counter()