import json
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "plant_disease_model.h5"
IMAGE_PATH = r"C:\FULL\PATH\TO\IMAGE.jpg"   # CHANGE

IMG_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH)

with open("class_names.json") as f:
    class_names = json.load(f)

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("Image not found")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
idx = np.argmax(pred)
conf = np.max(pred)

print("Disease:", class_names[idx])
print("Confidence:", round(conf * 100, 2), "%")