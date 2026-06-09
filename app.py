from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model(os.path.join(BASE_DIR, 'model', 'MobileNetV2.keras'))

CLASS_NAMES = [
    'Black Point',
    'Fusarium Foot Rot',
    'Healthy Leaf',
    'Leaf Blight',
    'Wheat Blast'
]

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return predicted_class, round(confidence, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']

        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction, confidence = predict_disease(filepath)
            image_path = url_for('static', filename=f'uploads/{filename}')

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == '__main__':
    app.run(debug=True)
