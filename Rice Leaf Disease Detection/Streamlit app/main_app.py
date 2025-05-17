import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

#Loading the Model
model=load_model('Rice_leaf_disease_detection.h5')

#Name of classes
Label_names=['Rice-Bacterial_leaf_blight' , 'Rice-Brown_spot' , 'Rice-Leaf_smut']

#Setting tilte of App
st.title("Rice leaf disease detection")
st.markdown("Upload an image of the rice leaf")

#Uploading the rice leaf image
rice_leaf_image=st.file_uploader("browse image", type='jpg')
submit=st.butoon("Predict disease")

#Predict button click code...
if submit:
    if rice_leaf_image is not None:
        file=np.asarry(bytearray(rice_leaf_image.read()),dtype=np.uint8)
        opencv=cv2.imdecode(file,1)
        
        #Displaying the image
        st.image(opencv,channels='BGR')
        st.write(opencv.shape)
 
        #Resizing the image
        opencv=cv2.resize(opencv,(256,256))
        #Convert image to 4 dimensions
        opencv.shape=(1,256,256,3)
 
        #Make predictions
        y_pred=model.predict(opencv)
        result=Label_names[np.argmax(y_pred)]
        st.title(str("This is"+result.split('-')[0]+"leaf with"+result.split('-')[1]))
