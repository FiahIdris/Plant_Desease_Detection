from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
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

# Load models
classifier = load_model('models/my_model.h5')
input_shape = classifier.layers[0].input_shape

@app.get("/")
def main_page():
    return "Hello, visit us on https://planting-projects2.herokuapp.com/docs"

@app.post("/detect-object")
async def detect_object(file: UploadFile = File(...)):
    contents = await file.read()
    detector = run_odt_and_draw_results(contents)
    res, im_png = cv2.imencode(".png", detector['image'])
    imgio =  BytesIO(im_png.tobytes())
 
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
        img_norm = img_batch/255
        prediction = classifier.predict(img_norm)
        index = prediction.argmax()
        predict_crop = CLASS_NAMES[index].split("__",1)[0]
        predict_class= CLASS_NAMES[index]  
        image2 = Image.open(f'contour_{index_to_predict}.png')
        imgio = BytesIO()
        image2.save(imgio, 'PNG')
        imgio.seek(0)
            
    totalLeaf = str(detector['totalLeaf'])
    return StreamingResponse(content=imgio, media_type="image/png",headers={'totalLeaf':totalLeaf,'crop':predict_crop,'class':predict_class})


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0')