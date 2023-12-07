from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Allow any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "./trainedModel.h5"
binary_model = load_model(model_path)

class Item(BaseModel):
    age: int
    gender: str  # Change 'sex' to 'gender' and update its type to str
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict")
def predict(item: Item):
    # Map categorical variables to numerical values
    gender_mapping = {'Female': 0,'Male': 1}

    # Convert input data to a NumPy array
    input_data = np.array([[
        item.age, gender_mapping.get(item.gender, 2), item.cp, item.trestbps, item.chol, item.fbs, item.restecg,
        item.thalach, item.exang, item.oldpeak, item.slope, item.ca, item.thal
    ]])

    # Make prediction using the loaded model
    prediction = binary_model.predict(input_data)

    # The output is a probability, you can convert it to a class (0 or 1) based on a threshold
    threshold = 0.5
    prediction_message = "Positive" if prediction[0, 0] > threshold else "Negative"

    return {"prediction": prediction_message}
    
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
