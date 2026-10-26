import os
import time
import cv2
import numpy as np
import tensorflow as tf
import serial
import serial.tools.list_ports
MODEL_PATH = "coin_model.keras"
CLASSES_PATH = "classes.txt"
print("[INFO] Loading model and classes...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
VALUE_MAP = {'1_rup': 1, '2_rup': 2, '5_rup': 5, '10_rup': 10, '20_rup': 20}
ser = None
for p in list(serial.tools.list_ports.comports()):
    if any(chip in p.description.lower() for chip in ["cp210", "ch340", "usb", "serial"]):
        try:
            ser = serial.Serial(p.device, 115200, timeout=1)
            time.sleep(2)
            print(f"[SUCCESS] Connected to ESP32 on {p.device}")
            break
        except Exception:
            pass
if not ser:
    print("[INFO] Running in Simulation Mode (ESP32 not connected).")
def check_coin_present(roi):
    """Checks if a coin is actually present in the box (not empty table)"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if np.std(gray) < 18.0:
        return False    
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    edges = cv2.Canny(blurred, 40, 110)
    return np.sum(edges > 0) > 400
def analyze_bimetallic(roi):
    """
    Rs. 10: Ring is GOLD, Center is SILVER.
    Rs. 20: Ring is SILVER, Center is GOLD.
    """
    h, w = roi.shape[:2]
    cx, cy = w // 2, h // 2
    r_outer = min(cx, cy)
    r_inner = int(r_outer * 0.35)
    center_roi = roi[cy - r_inner:cy + r_inner, cx - r_inner:cx + r_inner]
    mask_ring = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_ring, (cx, cy), int(r_outer * 0.88), 255, -1)
    cv2.circle(mask_ring, (cx, cy), int(r_outer * 0.65), 0, -1)
    ring_pixels = roi[mask_ring == 255]
    if center_roi.size == 0 or ring_pixels.size == 0:
        return "UNKNOWN", 0.0
    c_b, _, c_r = np.mean(center_roi, axis=(0, 1))
    r_b, _, r_r = np.mean(ring_pixels, axis=0)
    center_gold = float(c_r) - float(c_b)
    ring_gold = float(r_r) - float(r_b)
    bimetallic_diff = center_gold - ring_gold
    if bimetallic_diff > 4.0:
        return "20_rup", bimetallic_diff   
    elif bimetallic_diff < -4.0:
        return "10_rup", bimetallic_diff   
    else:
        return "UNIFORM", bimetallic_diff  
def measure_coin_radius(roi):
    """Measures the pixel radius of the coin to separate Rs. 1 and Rs. 2"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=50, param2=30, minRadius=40, maxRadius=115
    )
    if circles is not None:
        return int(circles[0][0][2])
    return None
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
BOX_SIZE = 240
SIZE_THRESHOLD_RADIUS = 82
print("\n" + "="*55)
print("     5-DENOMINATION BULLETPROOF COIN SORTER")
print("="*55)
print("Coins supported: Rs. 1 | Rs. 2 | Rs. 5 | Rs. 10 | Rs. 20")
print("Press SPACEBAR to confirm | 'q' to quit.\n")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    x1 = (w - BOX_SIZE) // 2
    y1 = (h - BOX_SIZE) // 2
    x2 = x1 + BOX_SIZE
    y2 = y1 + BOX_SIZE
    roi = frame[y1:y2, x1:x2].copy()
    has_coin = check_coin_present(roi)
    if not has_coin:
        box_color = (0, 165, 255)  # Orange
        status_str = "NO COIN PLACED -> Rs. 0"
        sub_str = "Waiting for coin... Place coin inside box"
        final_class = None
        coin_val = 0
        detected_radius = None
        bimetallic_type = "NONE"
    else:
        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized.astype("float32"), axis=0)
        preds = model.predict(img_array, verbose=0)[0]
        best_idx = np.argmax(preds)
        conf = preds[best_idx] * 100
        ai_class = class_names[best_idx]
        bimetallic_type, b_score = analyze_bimetallic(roi)
        detected_radius = measure_coin_radius(roi)
        final_class = ai_class
        if ai_class in ['10_rup', '20_rup']:
            if bimetallic_type == "20_rup":
                final_class = '20_rup'
            elif bimetallic_type == "10_rup":
                final_class = '10_rup'
        elif ai_class in ['1_rup', '2_rup']:
            if detected_radius is not None:
                if detected_radius >= SIZE_THRESHOLD_RADIUS:
                    final_class = '2_rup'
                else:
                    final_class = '1_rup'
        coin_val = VALUE_MAP.get(final_class, 0)
        box_color = (0, 255, 0)  # Green
        status_str = f"Coin: {final_class} -> Rs. {coin_val}"
        rad_text = f"Radius: {detected_radius}px" if detected_radius else "Radius: detecting..."
        sub_str = f"AI: {ai_class} | {rad_text} | Bimetallic: {bimetallic_type}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cx, cy = w // 2, h // 2
    cv2.circle(frame, (cx, cy), SIZE_THRESHOLD_RADIUS - 10, (100, 255, 100), 1)
    cv2.circle(frame, (cx, cy), SIZE_THRESHOLD_RADIUS + 15, (0, 255, 255), 1)
    cv2.putText(frame, status_str, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 2)
    cv2.putText(frame, sub_str, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
    cv2.putText(frame, "Press SPACEBAR to confirm & send | 'q' to quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Automated Coin Sorter - 5 Denominations", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 32:
        if not has_coin:
            print("[INFO] No coin placed! Please place a coin inside the box first.")
        else:
            print("\n" + "="*45)
            print("          FINAL DETECTION")
            print("="*45)
            print(f"  AI Prediction    : {ai_class} ({conf:.1f}%)")
            print(f"  Bimetallic Test  : {bimetallic_type} (score: {b_score:.1f})")
            print(f"  Radius Measured  : {detected_radius} px")
            print(f"  >> RESULT LOCKED : {final_class} -> Rs. {coin_val}")
            print("="*45)
            if ser and ser.is_open:
                ser.write(f"{coin_val}\n".encode())
                print(f"[SENT TO ESP32] -> {coin_val}")
            else:
                print(f"[SIMULATION] ESP32 received: Rs. {coin_val}")
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
if ser and ser.is_open:
    ser.close()