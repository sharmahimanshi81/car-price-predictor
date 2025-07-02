
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("quikr_car.csv")

# Data cleaning
df = df[df['Price'] != 'Ask For Price']
df['Price'] = df['Price'].str.replace(',', '', regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['kms_driven'] = df['kms_driven'].str.split().str.get(0)
df['kms_driven'] = df['kms_driven'].str.replace(',', '', regex=False)
df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
df.dropna(subset=['Price', 'year', 'kms_driven', 'fuel_type'], inplace=True)


df = df[df['fuel_type'].isin(['Petrol', 'Diesel'])]
df['name'] = df['name'].str.strip().str.title()
df['company'] = df['company'].str.strip().str.title()
df['fuel_type'] = df['fuel_type'].str.strip().str.title()

# Features and labels
X = df[['name', 'company', 'fuel_type', 'year', 'kms_driven']]
y = df['Price']

# Preprocessing
categorical_features = ['name', 'company', 'fuel_type']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Create model folder if not exist
os.makedirs("model", exist_ok=True)

# Save model
with open("model/pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ pipeline.pkl saved successfully!")
