import cv2

def list_cameras(max_index=10):
    available_cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera found at index {index}")
            available_cameras.append(index)
            cap.release()
        else:
            print(f"No camera at index {index}")
    return available_cameras

if __name__ == "__main__":
    cameras = list_cameras()
    if cameras:
        print(f"Available camera indices: {cameras}")
    else:
        print("No cameras found.")