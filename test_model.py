import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("pipeline.pkl", "rb"))  # No "model/" needed because we're already inside model/

# Test input
sample = pd.DataFrame([{
    "name": "Maruti Swift".strip().title(),
    "company": "Maruti".strip().title(),
    "fuel_type": "Petrol".strip().title(),
    "year": 2018,
    "kms_driven": 24000
}])


# Predict
prediction = model.predict(sample)
print("Predicted Price:", prediction[0])
