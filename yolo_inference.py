from ultralytics import YOLO

model = YOLO('yolov8x')

results = model.predict('input_videos/08fd33_4.mp4', save = True)
print(results[0]) # The first frame of video
print('##########################')
# Print all boxes detected in the first frame
for box in results[0].boxes:
  print(box)