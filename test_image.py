import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
MODEL_PATH = "coin_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
VALUE_MAP = {
    '1_rup': 1,
    '2_rup': 2,
    '5_rup': 5,
    '10_rup': 10,
    '20_rup': 20
}
def predict_coin(img_path):
    print(f"\nImage found: {img_path}")
    print("Running ML prediction...")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)[0]
    best_idx = np.argmax(predictions)
    confidence = predictions[best_idx] * 100
    predicted_class = class_names[best_idx]
    coin_val = VALUE_MAP.get(predicted_class, 0)
    print("-" * 35)
    print("          ML RESULT")
    print("-" * 35)
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.2f} %")
    print(f"Detected value: Rs. {coin_val}")
    print(f"OLED should display: Rs. {coin_val}")
    print("-" * 35)
if __name__ == "__main__":
    # Test on any image from your dataset (or provide your own image path)
    test_img = r"1_rup/design_1/coin (1).jpeg"
    predict_coin(test_img)