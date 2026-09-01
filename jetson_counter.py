"""
Full pipeline test: Camera -> TensorRT detection -> Zone counting
Combines live_detect_test.py (detection) with the ZoneCounter logic from
app_zone.py (counting) into one running script.

Since there's no belt right now, test this by holding a potato and sliding
it by hand through the blue zone rectangle drawn on screen.

Controls:
  s = start counting (also resets warm-up timer)
  x = stop/pause counting
  r = reset count to zero
  q = quit
"""

import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# ============================================================================
# CONFIG
# ============================================================================
ENGINE_PATH = "best.engine"
INPUT_SIZE = 640
CAMERA_INDEX = 0          # Fantech camera device index
CONF_THRESH = 0.65
STRICT_CONF_THRESH = 0.65  # extra bar for brand-new zone entries
NMS_IOU_THRESH = 0.55

# Zone as fraction of frame width/height (x1, y1, x2, y2), each 0..1
ZONE_X1_RATIO = 0.30
ZONE_Y1_RATIO = 0.30
ZONE_X2_RATIO = 0.70
ZONE_Y2_RATIO = 0.70

MAX_MATCH_DISTANCE = 100    # px, one-frame nearest-centroid matching gate
MAX_FRAMES_UNSEEN = 6
MIN_BOX_AREA_PX = 400
MAX_BOX_AREA_PX = 60000
WARMUP_FRAMES = 15

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ============================================================================
# TensorRT detector (same as live_detect_test.py)
# ============================================================================
def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


class TRTDetector:
    def __init__(self, engine_path):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.input_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
        self.output_shape = tuple(self.context.get_binding_shape(1))

        self.input_host = np.empty(self.input_shape, dtype=np.float32)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output_host = np.empty(self.output_shape, dtype=np.float32)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)
        self.bindings = [int(self.input_device), int(self.output_device)]
        self.stream = cuda.Stream()

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        pad_w = INPUT_SIZE - new_w
        pad_h = INPUT_SIZE - new_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img), scale, left, top

    def infer(self, input_tensor):
        np.copyto(self.input_host, input_tensor)
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()
        return self.output_host.copy()

    def postprocess(self, raw_output, scale, pad_x, pad_y, orig_w, orig_h):
        preds = raw_output[0].transpose(1, 0)  # (8400, 5)

        # Vectorized confidence filter (was a per-row Python loop before —
        # that loop over up to 8400 candidates every single frame was almost
        # certainly the real cause of the 2-3 FPS slowdown, not the camera).
        scores = preds[:, 4]
        mask = scores >= CONF_THRESH
        preds = preds[mask]
        scores = scores[mask]

        if preds.shape[0] == 0:
            return []

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        scores_list = scores.tolist()

        idxs = cv2.dnn.NMSBoxes(boxes, scores_list, CONF_THRESH, NMS_IOU_THRESH)
        results = []
        if len(idxs) > 0:
            for i in np.array(idxs).flatten():
                x, y, w, h = boxes[i]
                results.append(((x, y, x + w, y + h), scores_list[i]))
        return results

    def detect(self, frame):
        input_tensor, scale, pad_x, pad_y = self.preprocess(frame)
        raw_output = self.infer(input_tensor)
        h, w = frame.shape[:2]
        return self.postprocess(raw_output, scale, pad_x, pad_y, w, h)


# ============================================================================
# Zone counting logic (adapted from app_zone.py)
# ============================================================================
class ZoneObject:
    _next_id = 1

    def __init__(self, centroid, box, frame_idx):
        self.local_id = ZoneObject._next_id
        ZoneObject._next_id += 1
        self.centroid = centroid
        self.box = box
        self.frames_unseen = 0

    def update(self, centroid, box):
        self.centroid = centroid
        self.box = box
        self.frames_unseen = 0

    def mark_unseen(self):
        self.frames_unseen += 1


class ZoneCounter:
    def __init__(self, zone_xyxy):
        self.zone = zone_xyxy
        self.active_objects = []
        self.total_count = 0
        self.counting_enabled = False
        self.start_frame_idx = 0

    @staticmethod
    def _centroid_of(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _in_zone(self, c):
        zx1, zy1, zx2, zy2 = self.zone
        return zx1 <= c[0] <= zx2 and zy1 <= c[1] <= zy2

    def start(self, frame_idx):
        self.counting_enabled = True
        self.start_frame_idx = frame_idx

    def stop(self):
        self.counting_enabled = False

    def reset(self):
        self.total_count = 0
        self.active_objects = []

    def update(self, detections, frame_idx):
        zone_dets = []
        for box, conf in detections:
            x1, y1, x2, y2 = box
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < MIN_BOX_AREA_PX or area > MAX_BOX_AREA_PX:
                continue
            c = self._centroid_of(box)
            if self._in_zone(c):
                zone_dets.append((c, box, conf))

        matched_obj, matched_det = set(), set()
        if self.active_objects and zone_dets:
            from scipy.optimize import linear_sum_assignment
            cost = np.zeros((len(self.active_objects), len(zone_dets)), dtype=np.float32)
            for i, obj in enumerate(self.active_objects):
                ox, oy = obj.centroid
                for j, (c, box, conf) in enumerate(zone_dets):
                    dist = np.hypot(ox - c[0], oy - c[1])
                    cost[i, j] = dist if dist <= MAX_MATCH_DISTANCE else 1e6
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, cidx in zip(row_idx, col_idx):
                if cost[r, cidx] >= 1e6:
                    continue
                self.active_objects[r].update(zone_dets[cidx][0], zone_dets[cidx][1])
                matched_obj.add(r)
                matched_det.add(cidx)

        still_active = []
        for i, obj in enumerate(self.active_objects):
            if i not in matched_obj:
                obj.mark_unseen()
            if obj.frames_unseen <= MAX_FRAMES_UNSEEN:
                still_active.append(obj)
        self.active_objects = still_active

        for j, (c, box, conf) in enumerate(zone_dets):
            if j not in matched_det:
                if conf < STRICT_CONF_THRESH:
                    continue
                new_obj = ZoneObject(c, box, frame_idx)
                self.active_objects.append(new_obj)
                if self.counting_enabled and (frame_idx - self.start_frame_idx) > WARMUP_FRAMES:
                    self.total_count += 1

        return self.active_objects


# ============================================================================
# Main
# ============================================================================
def main():
    print("[INFO] Loading TensorRT engine...")
    detector = TRTDetector(ENGINE_PATH)

    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return
    w = int(cap.get(3))
    h = int(cap.get(4))
    print(f"[INFO] Resolution: {w}x{h}")

    zone_xyxy = (w * ZONE_X1_RATIO, h * ZONE_Y1_RATIO, w * ZONE_X2_RATIO, h * ZONE_Y2_RATIO)
    counter = ZoneCounter(zone_xyxy)

    print("[INFO] Controls: s=start  x=stop  r=reset  q=quit")

    frame_idx = 0
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break
        frame_idx += 1

        detections = detector.detect(frame)
        active_objects = counter.update(detections, frame_idx)

        # draw zone
        zx1, zy1, zx2, zy2 = map(int, zone_xyxy)
        overlay = frame.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)

        # draw detections
        for (x1, y1, x2, y2), conf in detections:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        # draw zone-tracked objects
        for obj in active_objects:
            x1, y1, x2, y2 = map(int, obj.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

        status = "RUNNING" if counter.counting_enabled else "STOPPED"
        status_color = (0, 255, 0) if counter.counting_enabled else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (320, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"COUNT: {counter.total_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        now = time.time()
        fps = 1.0 / (now - t_prev) if now > t_prev else 0.0
        t_prev = now
        cv2.putText(frame, f"fps: {fps:.1f}", (w - 140, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Potato Counter - Full Pipeline Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            counter.start(frame_idx)
            print(f"[INFO] Counting STARTED at frame {frame_idx}")
        elif key == ord("x"):
            counter.stop()
            print(f"[INFO] Counting STOPPED. Count so far: {counter.total_count}")
        elif key == ord("r"):
            counter.reset()
            print("[INFO] Count RESET to 0")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Final count: {counter.total_count}")


if __name__ == "__main__":
    main()