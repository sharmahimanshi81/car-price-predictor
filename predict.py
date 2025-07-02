import pickle
import pandas as pd

# Load model
with open("pipeline.pkl", "rb") as f:

    model = pickle.load(f)

# Example input (same format as training data)
data = pd.DataFrame([{
    'name': 'Maruti Swift',
    'company': 'Maruti',
    'fuel_type': 'Petrol',
    'year': 2014,
    'kms_driven': 40000
}])

# Predict
predicted_price = model.predict(data)
print(f"Predicted Price: ₹{int(predicted_price[0])}")
