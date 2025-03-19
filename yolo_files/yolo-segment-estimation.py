from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("video_clips/pull1.mp4", stream=True)  # predict on an image

for result in results:
    print(result.boxes)
    print('-'*100)
    print(result.masks)