import numpy as np
import sklearn 
import pickle
import cv2


haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
model_svm = pickle.load(open('./model/model_svm.pickle', mode='rb'))
pca_models = pickle.load(open('./model/pca_dict.pickle', mode='rb'))
models_pca = pca_models['pca']
mean_face_arr = pca_models['mean_face']


def faceRecognitionPipeline(filename, path=True):
    if path:
        # step-01: read image
        img = cv2.imread(filename)
    else:
        img = filename
    # step-02: convert into gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # step-03: crop the face (using haar cascade classifier)
    faces = haar.detectMultiScale(gray,1.5,3)
    predictions = []
    for x,y,w,h in faces:
        roi = gray[y:y+h, x:x+w]
    # step_04: normalization (0-1)
        roi = roi/255.0
    # step_05: resize images (100,100)
        if roi.shape[1]>100:
            roi_resize=cv2.resize(roi,(100,100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_CUBIC)
    # step_06: falttening (1x1000)
        roi_reshape = roi_resize.reshape(1,10000)
    # step_07: subtract with mean
        roi_mean = roi_reshape - mean_face_arr
    # step_08: get eigen image (apply roi mean to pca)
        eigen_image = models_pca.transform(roi_mean)
    # step_09: Eigen Image for visualisation
        eig_img = models_pca.inverse_transform(eigen_image)
    # step_10: pass to ml model (svm) and get predictions
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        print(results, prob_score)
    # step_11: generate report
        text = "%s : %d"%(results[0], prob_score_max*100)
    
    
        if results[0] == 'male':
            color = (255,0,255)
        else:
            color = (0, 255, 255)
        cv2.rectangle(img,(x,y), (x+w, y+h), color, 2)
        cv2.rectangle(img,(x,y-40), (x+w, y), color, -1)
        cv2.putText(img,text,(x,y), cv2.FONT_HERSHEY_PLAIN,1.6,(255,255,255),2)
    
        output = {
            'roi':roi,
            'eig_img': eig_img,
            'prediction_name':results[0],
            'score': prob_score_max
        }
    
        predictions.append(output)

    return img, predictions





    
