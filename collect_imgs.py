import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = [16]
dataset_size = 300

camera_index = 0  # Use the correct camera index found from the previous script
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

for j in number_of_classes:
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Error: Could not read frame or frame is empty.")
            continue

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 200
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Error: Could not read frame or frame is empty.")
            continue
        
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
        file_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(file_path, frame)
        print(f"Saved: {file_path}")
        
        counter += 1

cap.release()
cv2.destroyAllWindows()
