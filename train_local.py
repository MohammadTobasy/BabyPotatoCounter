import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load the pre-trained YOLO11 small model
    model = YOLO("yolo11s.pt") 

    # 2. Get the absolute path to your data.yaml
    dataset_path = os.path.abspath("TechnoSeedsbabyPotatoCounting-4/data.yaml")

    # 3. Start training on your RTX 3060 GPU with safe multiprocessing settings
    model.train(
        data=dataset_path,
        epochs=10,        
        imgsz=640,
        device=0,         # GPU
        workers=0         # Disables multiprocessing data loading to fully bypass Windows bugs
    )

    print("🎉 Training finished successfully on GPU!")
    print("Your 'best.pt' file is saved inside 'runs/detect/train/weights/'")