import os
import glob
import time
import cv2
import numpy as np
import serial
import serial.tools.list_ports
import tensorflow as tf
ports = list(serial.tools.list_ports.comports())
ser = None
for p in ports:
    if any(chip in p.description.lower() for chip in ["cp210", "ch340", "usb", "serial"]):
        try:
            ser = serial.Serial(p.device, 115200, timeout=1)
            time.sleep(2)
            print(f"[SUCCESS] Connected to ESP32 on {p.device}")
            break
        except:
            pass
print("\n[INFO] Loading trained coin model...")
model = tf.keras.models.load_model("coin_model.keras")
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"[INFO] Loaded Classes: {class_names}")
VALUE_MAP = {
    '1_rup': 1,
    '2_rup': 2,
    '5_rup': 5,
    '10_rup': 10,
    '20_rup': 20
}
print("\n" + "="*50)
print("RUNNING DIAGNOSTIC TEST ON ALL 5 COIN CLASSES...")
print("="*50)
for c_name in class_names:
    img_files = glob.glob(f"{c_name}/**/*.jpeg", recursive=True) + \
                glob.glob(f"{c_name}/**/*.jpg", recursive=True) + \
                glob.glob(f"{c_name}/**/*.png", recursive=True)
    if img_files:
        test_path = img_files[0]
        test_img = cv2.imread(test_path)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = cv2.resize(test_img, (224, 224))
        arr = np.expand_dims(test_img.astype("float32"), axis=0)
        preds = model.predict(arr, verbose=0)[0]
        pred_label = class_names[np.argmax(preds)]
        conf = np.max(preds) * 100
        print(f"Sample [{c_name:<6}] -> Predicted: {pred_label:<6} ({conf:.1f}%)")
print("="*50 + "\n")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
CROP_SIZE = 220 
print("HOW TO USE:")
print("1. Lay the coin flat on your table.")
print("2. Move the camera so the coin fits neatly inside the CIRCLE/BOX.")
print("3. Press SPACEBAR to identify and sort!")
print("4. Press 'q' to quit.\n")
last_sent_val = None
last_sent_time = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    x1 = cx - (CROP_SIZE // 2)
    y1 = cy - (CROP_SIZE // 2)
    x2 = x1 + CROP_SIZE
    y2 = y1 + CROP_SIZE
    coin_roi = frame[y1:y2, x1:x2].copy()
    roi_rgb = cv2.cvtColor(coin_roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (224, 224))
    img_array = np.expand_dims(roi_resized.astype("float32"), axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    best_idx = np.argmax(preds)
    conf = preds[best_idx] * 100
    detected_class = class_names[best_idx]
    coin_val = VALUE_MAP.get(detected_class, 0)
    reticle_color = (0, 255, 0) if conf >= 80 else (0, 220, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), reticle_color, 2)
    cv2.circle(frame, (cx, cy), CROP_SIZE // 2, reticle_color, 2)
    cv2.putText(frame, "ALIGN COIN HERE", (x1 + 25, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, reticle_color, 2)
    info_text = f"Live: {detected_class} ({conf:.1f}%) -> Rs. {coin_val}"
    cv2.putText(frame, info_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, reticle_color, 2)
    cv2.putText(frame, "Press SPACEBAR to confirm & send | 'q' to quit", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    preview_size = 150
    pip = cv2.resize(coin_roi, (preview_size, preview_size))
    frame[10:10+preview_size, w-preview_size-10:w-10] = pip
    cv2.rectangle(frame, (w-preview_size-10, 10), (w-10, 10+preview_size), (255, 255, 255), 2)
    cv2.putText(frame, "AI View", (w-preview_size-5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Automated Coin Sorting - Vision System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 32:
        print("\n" + "="*45)
        print("          DETECTED COIN BREAKDOWN")
        print("="*45)
        for i, name in enumerate(class_names):
            bar = "█" * int(preds[i] * 20)
            print(f"  {name:<7}: {preds[i]*100:6.2f}% | {bar}")
        print("-" * 45)
        print(f"  CONFIRMED RESULT : {detected_class} -> Rs. {coin_val}")
        print("="*45)
        if ser and ser.is_open:
            ser.write(f"{coin_val}\n".encode())
            print(f"[SENT TO ESP32] -> {coin_val}")
        else:
            print(f"[SIMULATION] ESP32 not connected. Sent: {coin_val}")
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
if ser and ser.is_open:
    ser.close()