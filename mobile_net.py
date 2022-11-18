import tensorflow as tf
import keras,os
import numpy as np,cv2
from keras.preprocessing.image import img_to_array,load_img
from keras.applications.mobilenet import MobileNet,preprocess_input,decode_predictions
import matplotlib.pyplot as plt

img_size = 224
mobile_model = MobileNet(weights="imagenet",include_top=True,
input_shape=(img_size,img_size,3),)
# print(mobile_model.summary())

def processing_img(img_file):
    img = load_img(img_file, target_size=(img_size,img_size))
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    pre_processInput = preprocess_input(img)    
    return pre_processInput
     



img_file = r"E:\CNN Project\TL\Gold_fish.jpg"
preprocessImgae = processing_img(img_file)
result = mobile_model.predict(preprocessImgae)
print(f"Prediction:- {decode_predictions(result,top=5)[0]}")
disp_img = cv2.imread(img_file)     
cv2.imshow("test image",cv2.resize(disp_img,(550,650)))
cv2.waitKey(0)
















