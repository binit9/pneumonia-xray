import os
import numpy as np
import random
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import warnings
import logging
warnings.filterwarnings('ignore')
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def process_image(image):
    image = image / 255
    image = cv2.resize(image, (224, 224))
    return image


def predict_image(model, path):
    img = cv2.imread(path)
    img = np.asarray(img)
    processed_img = process_image(img)
    processed_img = tf.expand_dims(processed_img, axis=0)
    # st.write("image dim = ", processed_img.shape)
    pred = model.predict(processed_img)
    return pred


# def load(model_path):
#     return load_model(model_path)


if __name__ == '__main__':
    repo = r'C:\Users\bbhagat\Documents\Datasets\pneumonia-xray'
    # repo = '/app'
    temp_dir = os.path.join(repo, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    model_dir = os.path.join(repo, 'models')
    # st.write(f'model_dir={model_dir}')
    # st.write(f'model_path={os.path.join(model_dir, model_name)}')

    st.title("Pneumonia Prediction From Chest X-Ray")

    with st.spinner("Please wait..."):
        model_name = st.selectbox("Please select a trained model", ['Inception', 'VGG_16', 'Mobilenet', 'Densenet_169', 'Stacked_Mobilenet_Densenet'])
        model_name = model_name.lower()+".h5"
        model_loaded = load_model(os.path.join(model_dir, model_name))
    buffer = st.file_uploader("Please upload X-ray image here")
    img = None
    temp_img_path = os.path.join(temp_dir, str(random.randint(0, 100)) + '.png')
    if buffer:
        img = Image.open(buffer)
        img.save(temp_img_path)
        st.image(img)
    # if img is not None:
        pred = predict_image(model_loaded, temp_img_path)[0][0]
        label = 'Pneumonia' if pred > 0.5 else 'Normal'
        pred = round(pred * 100) if label == 'Pneumonia' else 100 - round(pred * 100)
        st.write(f"CNN Model: {model_name[:-3].upper()}")
        st.write(f"Model Prediction : {label.upper()}")
        st.write(f"Prediction Confidence : {pred}%")
        st.success(f"Prediction Completed!")
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

