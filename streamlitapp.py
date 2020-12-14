import streamlit as st
import base64
import io
import requests
from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random
import onnxruntime as rt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# use file uploader object to recieve image
# Remember that this bytes object can be used only once
def bytesioObj_to_base64str(bytesObj):
    return base64.b64encode(bytesObj.read()).decode("utf-8")

# Image conversion functions
def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img

def PILImage_to_cv2(img):
    return np.asarray(img)

def ImgURL_to_base64str(url):
    return base64.b64encode(requests.get(url).content).decode("utf-8")

def drawboundingbox(img, boxes,pred_cls, rect_th=2, text_size=1, text_th=2):
    img = PILImage_to_cv2(img)
    class_color_dict = {}

    #initialize some random colors for each class for better looking bounding boxes
    for cat in pred_cls:
        class_color_dict[cat] = [random.randint(0, 255) for _ in range(3)]

    for i in range(len(boxes)):
        cv.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
                      (int(boxes[i][2]),int(boxes[i][3])),
                      color=class_color_dict[pred_cls[i]], thickness=rect_th)
        cv.putText(img,pred_cls[i], (int(boxes[i][0]), int(boxes[i][1])-5),  cv.FONT_HERSHEY_SIMPLEX, text_size, class_color_dict[pred_cls[i]],thickness=text_th)
    return img

def inference_detector(session, img, tresh=0.3):
    pred_classes, pred_boxes, pred_confidence = [], [], []
    h, w = img.shape[:2]
    # Image resize
    img = cv.resize(img, (416,416))
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    # Image normalization
    mean = np.float64(np.array([0,0,0]).reshape(1, -1))
    stdinv = 1 / np.float64(np.array([255.0,255.0,255.0]).reshape(1, -1))
    img = img.astype(np.float32)
    img = cv.subtract(img, mean, img)  
    img = cv.multiply(img, stdinv, img)  
    # Convert to [batch, c, h, w] shape
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    # Run model
    outputs = session.run(None, {'input': img})
    # Prepare results
    for box, cls in zip(outputs[0], outputs[1]):
        if box[-1] > tresh:
            pred_confidence.append(float(box[-1]))
            pred_classes.append(str(CLASSES[cls]))
            pred_boxes.append([int(box[0]*w/416), int(box[1]*h/416), int(box[2]*w/416), int(box[3]*h/416)])
    return {'boxes': pred_boxes, 
            'classes': pred_classes, 
            'confidence': pred_confidence}

@st.cache
def get_session(model_path):
    return rt.InferenceSession(model_path)

CLASSES = ('Cat', 'Raccoon', 'Dog', 'Fox', 'Person', 'Mouse', 'Porcupine', 
               'Human_hand', 'Bird', 'Rabbit', 'Skunk', 'Squirrel', 'Deer', 'Snake')
model_path = './tiny_model.quant2.fix.onnx'
sess = get_session(model_path)

st.markdown("<h1>Animal Detector App</h1><br>", unsafe_allow_html=True)

bytesObj = st.file_uploader("Choose an image file")

st.markdown("<center><h2>or</h2></center>", unsafe_allow_html=True)

url = st.text_input('Enter URL')

st.sidebar.title('Model parameters')
thresh = st.sidebar.slider(
    'Threshold:',
    .0, 1., (.5)
)

if bytesObj or url:
    # In streamlit we will get a bytesIO object from the file_uploader
    # and we convert it to base64str for our FastAPI
    if bytesObj:
        base64str = bytesioObj_to_base64str(bytesObj)

    elif url:
        base64str = ImgURL_to_base64str(url)

    # We will also create the image in PIL Image format using this base64 str
    # Will use this image to show in matplotlib in streamlit
    img = base64str_to_PILImage(base64str)
    img = np.asarray(img)

    # Model prediction
    result = inference_detector(sess, img.copy(), tresh=thresh)
    
    st.markdown("<center><h1>App Result</h1></center>", unsafe_allow_html=True)
    img = drawboundingbox(img, result['boxes'], result['classes'])
    # st.pyplot()
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    st.pyplot(fig)
    st.markdown("<center><h1>FastAPI Response</h1></center><br>", unsafe_allow_html=True)
    st.write(result)

