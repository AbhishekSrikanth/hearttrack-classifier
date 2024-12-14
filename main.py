# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import base64
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Load the pretrained VGG19 model
model = load_model('./models/model_vgg19.h5')

# Pydantic model for the request body
class ImageRequest(BaseModel):
    file: str  # base64 encoded image string

@app.post("/predict/")
async def predict(request: ImageRequest):
    # Decode the base64 image string
    img_data = io.BytesIO(base64.b64decode(request.file))
    img = Image.open(img_data)
    img = img.resize((224, 224))  # Resize image to 224x224 for VGG19

    # Convert to RGB if the image is not in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert image to numpy array and preprocess for VGG19
    img_array = np.array(img)

    if img_array.shape != (224, 224, 3):
        raise ValueError(f"Invalid image shape: {img_array.shape}. Expected shape (224, 224, 3).")
    
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)

    # Predict using the model
    preds = model.predict(img_array)

    return {"prediction": preds.tolist()[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
