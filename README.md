### Training Face Recognition Model


## Project Functioning

This project is a Machine Learning–based web application that predicts **Gender (Male/Female)** from a face image.

### How It Works

1. The user uploads an image through the web interface.  
2. The system detects the **face** using OpenCV Haar Cascade.  
3. The detected face is **cropped and preprocessed** (grayscale, resize, normalize).  
4. The processed image is converted into **Eigen Face features using PCA**.  
5. These features are passed to a **trained SVM Machine Learning model**.  
6. The model predicts the **Gender (Male/Female)**.  
7. The prediction result is displayed on the web page.  

### Processing Pipeline

Upload Image → Face Detection → Preprocessing → PCA (Eigen Faces) → SVM Model → Gender Prediction
