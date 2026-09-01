"""
Live detection test: Orbbec Gemini (RGB) + TensorRT engine (best.engine)
No counting/zone logic yet — this just draws boxes so you can visually
confirm the model still detects potatoes well through this camera, in your
room, before you're back at the belt.

Controls: press 'q' in the video window to quit.

Requires a monitor attached to the Jetson (uses cv2.imshow).
"""

import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

ENGINE_PATH = "best.engine"
INPUT_SIZE = 640
CONF_THRESH = 0.35      # start conservative; lower later if potatoes are missed
NMS_IOU_THRESH = 0.45

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


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
        """Letterbox-resize frame to INPUT_SIZE x INPUT_SIZE, keeping aspect
        ratio, padding with gray. Returns the model input tensor plus the
        scale/pad info needed to map boxes back to original frame coords."""
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
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # add batch dim

        return np.ascontiguousarray(img), scale, left, top

    def infer(self, input_tensor):
        np.copyto(self.input_host, input_tensor)
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()
        return self.output_host.copy()

    def postprocess(self, raw_output, scale, pad_x, pad_y, orig_w, orig_h):
        """raw_output shape: (1, 5, 8400) -> [cx, cy, w, h, conf] per anchor."""
        preds = raw_output[0].transpose(1, 0)  # -> (8400, 5)

        boxes = []
        scores = []
        for cx, cy, w, h, conf in preds:
            if conf < CONF_THRESH:
                continue
            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # x,y,w,h for NMSBoxes
            scores.append(float(conf))

        if not boxes:
            return []

        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_IOU_THRESH)
        results = []
        if len(idxs) > 0:
            for i in np.array(idxs).flatten():
                x, y, w, h = boxes[i]
                results.append(((x, y, x + w, y + h), scores[i]))
        return results

    def detect(self, frame):
        input_tensor, scale, pad_x, pad_y = self.preprocess(frame)
        raw_output = self.infer(input_tensor)
        h, w = frame.shape[:2]
        return self.postprocess(raw_output, scale, pad_x, pad_y, w, h)


def main():
    print("[INFO] Loading TensorRT engine...")
    detector = TRTDetector(ENGINE_PATH)

    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return

    print(f"[INFO] Actual resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print("[INFO] Press 'q' in the video window to quit.")

    t_prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        detections = detector.detect(frame)

        for (x1, y1, x2, y2), conf in detections:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        now = time.time()
        fps = 1.0 / (now - t_prev) if now > t_prev else 0.0
        t_prev = now
        cv2.putText(frame, f"detections: {len(detections)}  fps: {fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Live Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()