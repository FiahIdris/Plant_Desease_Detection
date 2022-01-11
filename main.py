from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn
from utils import run_odt_and_draw_results
import cv2
from tensorflow.keras.preprocessing import image

app = FastAPI()
CROP_NAMES = ['Apple','Corn','Grape','Pepper','Potato','Strawberry','Tomato']
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
PLANT_NAMES = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas','mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate','banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple','orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']

class Plant(BaseModel):
    N: float
    P: float
    K: float
    temp: float
    humidity: float
    pH: float
    rainfall: float
    
# Load models
classifier = load_model('models/my_model.h5')
recommender = load_model('models/recommender_system.h5')
input_shape = classifier.layers[0].input_shape

@app.get("/")
def main_page():
    return "Hello, visit us on https://planting-project.herokuapp.com/docs"

@app.get("/class-names")
def get_available_class():
    return {"crop_names":CROP_NAMES, "class_names": CLASS_NAMES,}

@app.post("/detect-object")
async def detect_object(file: UploadFile = File(...)):
    contents = await file.read()
    detector = run_odt_and_draw_results(contents)
    res, im_png = cv2.imencode(".png", detector['image'])

    predict_crop = ''
    predict_class= ''
    
    if detector['totalLeaf'] == 1:
        # read image contain
        pil_image = Image.open(BytesIO(contents))
        # resize image to expected input shape
        pil_image = pil_image.resize((input_shape[1], input_shape[2]))

        # convert imgae to numpy format
        numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))
    
        # scale data
        numpy_image = numpy_image / 255.0
        # generate prediction
        prediction_array = np.array([numpy_image])
        prediction = classifier.predict(prediction_array)
        index = prediction.argmax()
        predict_crop = CLASS_NAMES[index].split("__",1)[0]
        predict_class= CLASS_NAMES[index]
        
    if detector['totalLeaf'] > 1:
        index_to_predict = np.array(detector['label']).argmax() +1
        img= image.load_img(f'contour_{index_to_predict}.png', target_size=(256,256))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_batch = img_batch/255
        prediction = classifier.predict(img_batch)
        index = prediction.argmax()
        predict_crop = CLASS_NAMES[index].split("__",1)[0]
        predict_class= CLASS_NAMES[index]
   
            
    totalLeaf = str(detector['totalLeaf'])
    return StreamingResponse(content=BytesIO(im_png.tobytes()), media_type="image/png",headers={'totalLeaf':totalLeaf,'crop':predict_crop,'class':predict_class})



@app.post("/predict-plant")
async def predict_plant(plant:Plant):
    prediction = recommender.predict([[plant.N, plant.P, plant.K, plant.temp, plant.humidity, plant.pH, plant.rainfall]])
    index = prediction[0].argmax()
    confidence = np.max(prediction[0])
    return {
        'prediction_class': PLANT_NAMES[index],
        'confidence': np.round(float(confidence),4),
    }

@app.post("/predict-leaf-disease")
async def predict_leaf_disease(file: UploadFile = File(...)):
    # read image contain
    contents = await file.read()
    pil_image = Image.open(BytesIO(contents))
    # resize image to expected input shape
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))

    # convert imgae to numpy format
    numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))
   
    # scale data
    numpy_image = numpy_image / 255.0
    # generate prediction
    prediction_array = np.array([numpy_image])
    prediction = classifier.predict(prediction_array)
    index = prediction.argmax()
    confidence = np.max(prediction)
    return {
        'prediction_crop' :CLASS_NAMES[index].split("__",1)[0],
        'prediction_class': CLASS_NAMES[index],
        'confidence': np.round(float(confidence),4),
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0')