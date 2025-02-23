from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# Define the request body structure
class FeaturesInput(BaseModel):
    features: list[float]  # Expecting a list of floating-point numbers

# Initialize FastAPI app
app = FastAPI()

# Load trained model
model = pickle.load(open("../saved_model/sales_prediction.sav", "rb"))

@app.post("/predict/")
def predict(input_data: FeaturesInput):
    try:
        # Convert input list into a NumPy array & reshape it for prediction
        features_array = np.array(input_data.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)
        return {"predicted_sales": float(prediction[0])}
    
    except Exception as e:
        return {"error": str(e)}
