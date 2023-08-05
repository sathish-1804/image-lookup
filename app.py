import numpy as np
import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO
from fastapi import FastAPI

app = FastAPI()

# Load the pre-trained TFLite model
model_path = "model.tflite"
interpreter = tf.lite.Interpreter(model_path)

# Function to preprocess the input image
def preprocess_image(image):
    input_shape = interpreter.get_input_details()[0]['shape']
    image = image.convert("RGB")
    image = image.resize((input_shape[1], input_shape[2]))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    input_data = image_array.astype(np.uint8)
    input_data = input_data[np.newaxis, ...]
    return input_data

# Function to predict the class
def predict_class_from_url(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            input_data = preprocess_image(image)
            
            # Run inference
            interpreter.allocate_tensors()
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            
            # Post-process the output
            labels = ['smartwatch', 'smartphone', 'laptop', 'headphones']
            predicted_class = labels[np.argmax(output_data)]
            
            return predicted_class
        else:
            return "Failed to fetch image from URL"
    except Exception as e:
        return str(e)

@app.get("/")
async def read_root():
    return {"message": "Image Classifier API"}

@app.get("/predict/")
async def predict_image_class(image_url: str):
    predicted_class = predict_class_from_url(image_url)
    return {"predicted_class": predicted_class}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)