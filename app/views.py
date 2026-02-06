import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request
import matplotlib.image as mating

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
        pred_image, prediction = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}', pred_image)
        print(prediction)
        report = []

        for i, obj in enumerate(prediction):
            gray_image = obj['roi'] 
            eigen_image = obj['eig_img'].reshape(100,100)
            gender_name = obj['prediction_name']
            score = round(obj['score']*100,2)

            gray_image_name = f'roi_{i}.jpg'
            eig_image_name = f'eigen_{i}.jpg'
            mating.imsave(f'./static/predict/{gray_image_name}', gray_image, cmap='gray')
            mating.imsave(f'./static/predict/{eig_image_name}', eigen_image, cmap='gray')

            report.append([gray_image_name, eig_image_name, gender_name, score])
    
        return render_template('gender.html', fileupload=True, report=report) #GET
    return render_template('gender.html',  fileupload= False) #POST

