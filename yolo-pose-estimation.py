from ultralytics import YOLO

# load a model
model = YOLO("yolo11n-pose.pt") # pretrain model

# predict with the model
results = model.track(source="video_clips/fulltoss.mp4",
                    show=True,
                    save=True) # predict of an image

# Access the results
for result in results:
    print(result.boxes)
    id = result.boxes.id
    xyxy = result.boxes.xyxy
    print(id, xyxy)

