import tensorflow as tf
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from keras.utils import load_img, img_to_array 
import numpy as np
from keras import preprocessing
from keras.models import load_model,Sequential
from keras.activations import softmax
from keras_applications.resnet50 import preprocess_input
import os
import h5py
import streamlit as st
st.header("Indian Sign Language Recognizor")
def main():
    file_uploaded=st.file_uploader("Choose the file",type=['jpg','png','jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)
        
def predict_class(image):
    classifier_model= load_model(r'E:\\TF\\streamlit_app\\CNN-ISL.h5')
    shape =((50,50,3))
    model = Sequential([hub.KerasLayer(classifier_model,input_shape=shape)])
    test_img = image.resize((50,50))
    test_img = img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    predictions = model.predict(test_img)
    predictions_list = list(predictions[0])
    class_names = ['1','2','3' ,'4', '5', '6' ,'7' ,'8' ,'9' ,'A' ,'B' ,'C' ,'D' ,'E' ,'F', 'G', 'H', 'I','J', 'K' ,'L' ,'M', 'N', 'O' ,'P' ,'Q', 'R' ,'S', 'T', 'U' ,'V' ,'W' ,'X' ,'Y' ,'Z']
    n = 0 
    for i in predictions_list:
        if i == 1.0:
            n = predictions_list.index(i)
    result ="The given symbol is predicted as {}".format( class_names[n] )
    return result

if __name__ == "__main__":
    main()

st.write("For best results please use the below given dataset for prediction")    
with open("Test_data.zip", "rb") as fp:
    btn = st.download_button(
        label="Download Test_data.ZIP",
        data=fp,
        file_name="test_Data.zip",
        mime="streamlit_app/zip"
    )
    
    
