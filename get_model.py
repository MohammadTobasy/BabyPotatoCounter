from roboflow import Roboflow

# Connect to Roboflow and download the model weights directly
rf = Roboflow(api_key="3SYbmdSFsNKkSeOu6bXd") 
project = rf.workspace("tobasys").project("technoseedsbabypotatocounting")
dataset = project.version(4).download("yolov8")

print("🎉 Success! The weights folder is fully downloaded.")