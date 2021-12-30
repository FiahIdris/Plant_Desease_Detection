from typing import Optional
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None
    
# Load models
classifier = load_model('my_model.h5')
input_shape = classifier.layers[0].input_shape

@app.get("/")
def read_root():
    return {"Hello" : "World"}

@app.get("/items/{item_id}")
def read_item(item_id:int, q:Optional[str]=None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item:Item):
    return {"item_name":item.name, "item_id":item_id}

CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
def read_file_as_img(data):
    
    datagen = ImageDataGenerator(rescale = 1. / 255.0)
   
    
    img = Image.open(BytesIO(data))
    file = datagen.fit(img)
    print("=======")
    print(file)
    img = img.resize((256,256),Image.ANTIALIAS)
    image = np.expand_dims(np.array(img), axis=0)
    return image/255

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
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
        'prediction': CLASS_NAMES[index],
        'confidence': np.round(float(confidence),4),
    }

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)