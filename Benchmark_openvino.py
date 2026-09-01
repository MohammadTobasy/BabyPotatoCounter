"""
Benchmark script — measures inference FPS of the OpenVINO-exported model
on this laptop, using a dummy image (no camera yet, isolate one variable
at a time, same approach we used on the Jetson).
"""

import time
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "best_openvino_model/"
IMG_SIZE = 640
NUM_WARMUP = 10
NUM_RUNS = 50


def main():
    print(f"[INFO] Loading OpenVINO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    dummy_image = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    print(f"[INFO] Warming up ({NUM_WARMUP} runs)...")
    for _ in range(NUM_WARMUP):
        model.predict(source=dummy_image, imgsz=IMG_SIZE, verbose=False)

    print(f"[INFO] Benchmarking ({NUM_RUNS} runs)...")
    t_start = time.time()
    for _ in range(NUM_RUNS):
        model.predict(source=dummy_image, imgsz=IMG_SIZE, verbose=False)
    elapsed = time.time() - t_start

    fps = NUM_RUNS / elapsed
    ms_per_frame = (elapsed / NUM_RUNS) * 1000

    print("=" * 50)
    print(f"[RESULT] {NUM_RUNS} runs in {elapsed:.2f}s")
    print(f"[RESULT] {fps:.1f} FPS  ({ms_per_frame:.1f} ms/frame)")
    print("=" * 50)


if __name__ == "__main__":
    main()