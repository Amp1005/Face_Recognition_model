import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request


UPLOAD_FOLDER = 'static/upload'



def index():
    return render_template('index.html')


def app():
    return render_template('app.html')


def genderapp():
    if request.method == 'POST':
        # Handle POST request logic here
        f = request.files['image_name']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)

         
    return render_template('gender.html')