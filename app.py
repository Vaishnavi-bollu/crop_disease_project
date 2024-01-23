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
             col4, col5 = stl.columns([1, 1])
             # 0 Wheat foot rot
             if label == "0 Wheat foot rot\n":
                     stl.success("The image is classified as Wheat foot rot")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )   
             #1
             if label == "1 wheat brown rust \n":
                     stl.success("The image is classified as  Wheat Brown Rust ")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )   
             #2
             if label == "2 wheat flag must\n":
                     stl.success("The image is classified as Wheat Flag Must")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )   
             #3
             if label == "3 wheat brown rust\n":
                     stl.success("The image is classified as Wheat Brown Rust")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )   
             #4
             if label == "4 wheat flag must\n":
                     stl.success("The image is classified as wheat flag must")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )
               #5
             if label == " 5 wheat leaf blotch\n":
                     stl.success("The image is classified as Wheat Leaf Blotch")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )
               #6
             if label == "6 wheat loose must\n":
                     stl.success("The image is classified as Wheat loose must")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )
               #7
             if label == "7 wheat stripe rust\n":
                     stl.success("The image is classified as Wheat stripe rust")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #8
             if label == "8 Sorghum grain mould\n":
                     stl.success("The image is classified as Sorghum grain mould")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )
               #9
             if label == "9 sorghum charcoal rot\n":
                     stl.success("The image is classified as sorghum charcoal rot")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )
               #10
                       
             if label == "10 sorghum downy mildew\n":
                     stl.success("The image is classified as sorghum downy mildew")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

             #11          
             if label == "11 sorghum of erogt\n":
                     stl.success("The image is classified as sorghum of erogt")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )                                    

               #12
             if label == "12 groundnut early leaf spot\n":
                     stl.success("The image is classified as  Groundnut early leaf spot")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #13
             if label == "13 ground nut late leaf spot\n":
                     stl.success("The image is classified as Ground nut late leaf spot")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )
               #14
             if label == "14 ground nut rust\n":
                     stl.success("The image is classified as Ground nut rust")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )
               #15
             if label == "15 groundnut steam rot\n":
                     stl.success("The image is classified as Groundnut steam rot")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #16
             if label == "16 ground nut bud necrosis\n":
                     stl.success("The image is classified as Ground nut bud necrosis")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )
               #17
             if label == "17 groundnut alternatia leaf spot\n":
                     stl.success("The image is classified as Groundnut alternatia leaf spot")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #18
             if label == "18 chilli bacterial leaf\n":
                     stl.success("The image is classified as Chilli bacterial leaf")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )
               #19
             if label == "19 chilli cercopora leaf spot\n":
                     stl.success("The image is classified as Chilli cercopora leaf spot")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #20
             if label == "20 chilli powdery mildew\n":
                     stl.success("The image is classified as Chilli powdery mildew")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #21
             if label == "21 chilli damping off\n":
                     stl.success("The image is classified as Chilli damping off")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #22
             if label == "22 chilli fruit rot and die back\n":
                     stl.success("The image is classified as Chilli fruit rot and die back")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #23
             if label == "23 sugarcane grassy shoot\n":
                     stl.success("The image is classified as Sugarcane grassy shoot")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #24
             if label == "24 sugarcane leaf scald disease\n":
                     stl.success("The image is classified as Sugarcane leaf scald disease")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #25
             if label == "25 sugarcane red rot\n":
                     stl.success("The image is classified as Sugarcane red rot")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #26
             if label == "26 sugarcane red strooped disease\n":
                     stl.success("The image is classified as Sugarcane red strooped disease")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #27
             if label == "27 sugarcane smut\n":
                     stl.success("The image is classified as Sugarcane smut")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )                  

               #28
             if label == "28 sugarcane wilt\n":
                     stl.success("The image is classified as Sugarcane wilt")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #29
             if label == "29 cotton boll root\n":
                     stl.success("The image is classified as Cotton boll root")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #30
             if label == "30 cotton leaf blight\n":
                     stl.success("The image is classified as Cotton leaf blight")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )


               #31
             if label == "31 cotton root rot\n":
                     stl.success("The image is classified as Cotton root rot")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #32
             if label == "32 cotton wilt\n":
                     stl.success("The image is classified as Cotton wilt")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #33
             if label == "33 jute stem rot\n":
                     stl.success("The image is classified as Jute stem rot")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #34
             if label == "34 jute black band\n":
                     stl.success("The image is classified as Jute Black Band")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               #35
             if label == "35 jute tip blight\n":
                     stl.success("The image is classified as Jute Tip Bligh")
                     with col4:
                       stl.image("sgd goals/12.png", use_column_width=True )
                       stl.image("sgd goals/15.png", use_column_width=True )
                     with col5:
                       stl.image("sgd goals/2.png", use_column_width=True )
                       stl.image("sgd goals/3.png", use_column_width=True )

               


        with col3:
             stl.info("information related to Corp Diseae Damage")    


