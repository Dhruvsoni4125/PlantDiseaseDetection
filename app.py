import json
import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
import h5py
from flask import Flask, render_template, request, Response

app = Flask(__name__)

IMG_SIZE = 224
CONF_THRESHOLD = 0.75

# ================= LOAD MODEL & CLASSES =================
def _remove_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            _remove_quantization_config(value)
    elif isinstance(obj, list):
        for item in obj:
            _remove_quantization_config(item)


def load_model_compat(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except TypeError as exc:
        if "quantization_config" not in str(exc):
            raise

        compat_path = os.path.splitext(model_path)[0] + "_compat.h5"
        if not os.path.exists(compat_path):
            shutil.copy2(model_path, compat_path)
            with h5py.File(compat_path, "r+") as model_file:
                raw_config = model_file.attrs.get("model_config")
                if raw_config is None:
                    raise
                if isinstance(raw_config, bytes):
                    config_text = raw_config.decode("utf-8")
                else:
                    config_text = raw_config

                config = json.loads(config_text)
                _remove_quantization_config(config)
                model_file.attrs.modify("model_config", json.dumps(config).encode("utf-8"))

        return tf.keras.models.load_model(compat_path, compile=False)


model = load_model_compat("plant_disease_model.h5")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

camera = cv2.VideoCapture(0)


def _clean_label_name(label):
    return label.replace("_", " ").replace("  ", " ").strip()


def get_cure_methods(predicted_label):
    if not predicted_label or predicted_label == "No leaf detected":
        return [
            "Retake a clear, close leaf image in good lighting for a reliable diagnosis.",
            "Ensure the leaf occupies most of the frame and is in focus.",
        ]

    disease_name = predicted_label.split("___", 1)[-1]
    disease_key = disease_name.lower()

    if "healthy" in disease_key:
        return [
            "No treatment required. Keep routine irrigation and balanced nutrition.",
            "Continue preventive monitoring and remove any damaged plant parts early.",
            "Avoid overwatering and maintain proper spacing for airflow.",
        ]

    if "bacterial" in disease_key:
        return [
            "Remove and destroy infected leaves to reduce spread.",
            "Avoid overhead watering; water at soil level in the morning.",
            "Use copper-based bactericide as per label guidance.",
            "Disinfect tools between plants and improve air circulation.",
        ]

    if "powdery_mildew" in disease_key:
        return [
            "Prune dense foliage and improve sunlight/air movement around plants.",
            "Apply sulfur or potassium bicarbonate fungicide as directed.",
            "Avoid excess nitrogen fertilizer that promotes soft, susceptible growth.",
        ]

    if "rust" in disease_key:
        return [
            "Remove infected leaves and nearby plant debris immediately.",
            "Apply a registered protective fungicide at regular intervals.",
            "Keep foliage dry and reduce humidity around plants.",
        ]

    if "virus" in disease_key or "mosaic" in disease_key or "curl" in disease_key:
        return [
            "There is no direct cure; remove severely infected plants promptly.",
            "Control vectors (whiteflies/aphids) using sticky traps and approved insecticides.",
            "Use resistant varieties and keep the field weed-free.",
        ]

    if "blight" in disease_key or "leaf_spot" in disease_key or "leaf_mold" in disease_key or "scab" in disease_key or "rot" in disease_key or "measles" in disease_key:
        return [
            "Remove infected leaves/fruits and dispose away from the field.",
            "Apply a recommended fungicide and rotate active ingredients.",
            "Improve drainage and avoid wet leaves for long durations.",
            "Follow crop rotation and field sanitation practices.",
        ]

    return [
        "Isolate affected plants and remove visibly infected tissue.",
        "Apply crop-specific fungicide/bactericide based on local extension advice.",
        "Maintain spacing, sanitation, and irrigation control to limit spread.",
    ]

# ================= PREPROCESS =================
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ================= PREDICT =================
def predict_image(img):
    img = preprocess(img)
    preds = model.predict(img, verbose=0)
    conf = float(np.max(preds))
    idx = int(np.argmax(preds))

    if conf < CONF_THRESHOLD:
        return "No leaf detected", 0.0

    return class_names[idx], round(conf * 100, 2)

# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    cure_methods = []
    disease_display_name = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", error="No selected file")

        image_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return render_template("index.html", error="Invalid image")

        result, confidence = predict_image(img)
        cure_methods = get_cure_methods(result)
        if result and result != "No leaf detected":
            disease_display_name = _clean_label_name(result.split("___", 1)[-1])

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        cure_methods=cure_methods,
        disease_display_name=disease_display_name
    )

# ================= LIVE CAMERA =================
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        label, conf = predict_image(frame)
        color = (0, 255, 0) if conf > 0 else (0, 0, 255)

        cv2.putText(frame, f"{label} ({conf}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)

