import cv2
import os

# 1. Define strict paths
VIDEO_NAME = "8.mp4"
INPUT_VIDEO_PATH = os.path.join("data", "DataSetS25U", VIDEO_NAME)
OUTPUT_FOLDER = os.path.join("data", "potato_images")

print(f"Checking for video file at: {os.path.abspath(INPUT_VIDEO_PATH)}")

# 2. Check if file physically exists before opening
if not os.path.exists(INPUT_VIDEO_PATH):
    print("❌ ERROR: The video file does not exist at that path!")
    print("Please check that your folder is named 'DataSetS25U' exactly (case-sensitive) and contains '8.mp4'.")
else:
    print("✅ Video file found! Attempting to open...")
    
    # Create the output folder if it's missing
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    
    # Check if OpenCV can actually decode the video format
    if not cap.isOpened():
        print("❌ ERROR: OpenCV could not open or decode the video file.")
        print("This usually means the file path is broken, the file is corrupted, or Windows is missing the MP4 codec.")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"🎬 Video opened successfully! Total frames: {total_frames} | FPS: {fps:.2f}")
        
        count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save every 15th frame (adjust this if you want more/fewer images)
            if count % 15 == 0:
                image_name = os.path.join(OUTPUT_FOLDER, f"potato_frame_{saved_count}.jpg")
                cv2.imwrite(image_name, frame)
                saved_count += 1
                
            count += 1

        cap.release()
        print("\n--- Extraction Finished! ---")
        print(f"🎉 Successfully saved {saved_count} photos inside: '{OUTPUT_FOLDER}'")