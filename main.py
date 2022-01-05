from typing import Optional
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn
import os

app = FastAPI()
CROP_NAMES = ['Apple','Corn','Grape','Pepper','Potato','Strawberry','Tomato']
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# class Crop(BaseModel):
#     name: str
#     price: float
#     is_offer: Optional[bool] = None
    
# Load models
classifier = load_model('my_model.h5')
input_shape = classifier.layers[0].input_shape

@app.get("/")
def main_page():
    return "Hello, visit us on https://planting-project.herokuapp.com/docs"

@app.get("/class-names")
def get_available_class():
    return {"crop_names":CROP_NAMES, "class_names": CLASS_NAMES,}


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item:Item):
#     return {"item_name":item.name, "item_id":item_id}

@app.post("/predict")
async def predict_new_image(file: UploadFile = File(...)):
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