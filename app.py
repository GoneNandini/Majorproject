from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
import pickle

app = Flask(__name__)

# Load the trained model
model_path = r"C:\Users\nandi\OneDrive\Desktop\Dhrma\currency\vgg-modelDhrma.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define class names
class_names = {0: "Fake Currency", 1: "Real Currency"}

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file uploaded')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction='No file selected')
        file_path = os.path.join("static", file.filename)
        file.save(file_path)
        
        # Process the image and predict
        img = process_image(file_path)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        result = class_names[predicted_class]
        
        return render_template('index.html', prediction=result, image_path=file_path)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

