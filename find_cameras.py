import cv2

print("Scanning for connected cameras...")
for idx in range(4):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"[FOUND] Camera working at Index: {idx}")
        cap.release()

print("Scan complete.")