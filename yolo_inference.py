from ultralytics import YOLO
model = YOLO('models/best.pt')
results = model.predict('INPUT_VIDEOS/08fd33_4.mp4',save = True) # ssaves the video 

print(results[0])
print('====================================')
for box in results[0].boxes:
    print(box)

