import os
import tensorflow as tf
if os.path.exists("classes.txt"):
    with open("classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"Classes in classes.txt ({len(classes)}): {classes}")
else:
    print("[ERROR] classes.txt not found!")
if os.path.exists("coin_model.keras"):
    model = tf.keras.models.load_model("coin_model.keras")
    num_outputs = model.output_shape[-1]
    print(f"Model output neurons: {num_outputs}")
    if num_outputs != 5: 
        print(f"\n[PROBLEM FOUND] The model was only trained on {num_outputs} classes, NOT 5!")
    else:
        print("\n[OK] Model has 5 output classes.")
else:
    print("[ERROR] coin_model.keras not found!")
print("\nImage count in each folder:")
for folder in ['1_rup', '2_rup', '5_rup', '10_rup', '20_rup']:
    if os.path.exists(folder):
        count = sum([len(files) for r, d, files in os.walk(folder) if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files)])
        print(f" - {folder:<8}: {count} images")
    else:
        print(f" - {folder:<8}: [FOLDER MISSING!]")