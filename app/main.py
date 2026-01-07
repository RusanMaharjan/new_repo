from fastapi import FastAPI
from app.schema import EmployeeTerm
import pandas as pd
from app.model import load_model, features

app = FastAPI()

model = load_model()

@app.post("/predict/")
def predict(data: EmployeeTerm):
    input_data = pd.DataFrame([
        data.dict()
    ], columns=features)

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    return {
        "Predicted Termination": int(prediction),
        "Status": "Terminated" if prediction == 1 else "Active",
        "Probability": {
            "Active": prob[0],
            "Terminated": prob[1]
        }
    }