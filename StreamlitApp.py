import streamlit as st
# import torch
# from torchvision import transforms
from PIL import Image,ImageDraw
import numpy as np
from ultralytics import YOLO

#Load Model
# model = torch.load('models/best_80Epochs_20000Dataset.pt')
# model = torch.hub.load('ultralytics/yolov8', 'custom', path='models/best_80Epochs_20000Dataset.pt')
model = YOLO(r'D:\AiRotor\Wind Turbine analysis\windTurbineAnalysis\models\best_40Epochs_20000Dataset.pt')

#define Predict Image
def predict_img(image):

    # Inference
    results = model.predict(source=image)
    return results[0]        

def draw_boxes(image, boxes,confidences, classes,names):
    draw = ImageDraw.Draw(image)

    for index, box in enumerate(boxes):
        if len(box) >= 4:  # Ensure there are enough values
            x1, y1, x2, y2 = box[:4]
            # print(x1,x2,y1,y2)
            if confidences[index] > 0.2:
                draw.rectangle([x1, y1, x2, y2], width=2)
                class_name = names[classes[index]]  # Get the class name
                draw.text((x1, y1), f'{class_name}: {confidences[index]:.2f}', fill="red")
    
    return image

        
#Streamlit App
st.title("Wind Turbine Defects Detection")

uploaded_file = st.file_uploader("Upload an image to check for defects..",type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    boxes = []
    confidences = []
    classes = []
    #Run Prediction
    results = predict_img(image)
    for result in results:
        boxes.extend(result.boxes.xyxy.tolist())  # Get boxes in format (x1, y1, x2, y2, confidence, class)
        confidences.extend(result.boxes.conf.tolist())
        classes.extend(result.boxes.cls.tolist())
    print("boxes Are hererererer :",boxes)    
    output_image = draw_boxes(image.copy(),boxes,confidences,classes,results.names)

    st.image(output_image, caption='Output Image', use_column_width=True)

    #Display detected defects and confidence values
    st.write("Detected Defects and confidence values:")
    for index, box in enumerate(boxes):
        if len(box) >= 4:  # Ensure there are enough values
            x1, y1, x2, y2 = box[:4]
            # print(x1,x2,y1,y2)
            if confidences[index] > 0.2:
                class_name = results.names[classes[index]]  # Get the class name
                st.write(f'{class_name}: {confidences[index]:.2f}')