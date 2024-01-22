from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from dotenv import load_dotenv 
import os
import openai
import streamlit as stl 

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def crop_detection(img):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("converted_keras/keras_model.h5", compile=False)

    # Load the labels
    class_names = open("converted_keras/labels.txt", "r") .readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Replace this with the path to your image
    image = img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    #print("Class:", class_name[2:], end="")
    #print("Confidence Score:", confidence_score)

    return class_name, confidence_score

stl.set_page_config(layout='wide')

stl.tittle=("Crop_Disease_Detection")

input_img = stl.file_uploader("enter your imagee", type =['jpg', 'png', 'jpeg'])

if input_img is not None:
    if stl.button("Classify"):

        col1, col2, col3 = stl.columns([1,1,1])
      
        with col1:
             stl.info("your uploaded image")
             stl.image(input_img, use_column_width=True)

        with col2:
             stl.info("Your result")
             image_file = Image.open(input_img)
             label, confidence_score = crop_detection(image_file)
             stl.write(label)
             stl.write(confidence_score)
             col4, col5 = stl.columns([1, 1]) 


        with col3:
             stl.info("information related to Corp Diseae Damage")    


