import streamlit as st
import json
import pickle
import numpy as np

# ----------------- LOAD MODEL + COLUMNS -----------------
@st.cache_resource
def load_artifacts():
    with open("columns (2).json", "r") as f:
        data_columns = json.load(f)["data_columns"]

    with open("banglore_home_prices_model (2).pickle", "rb") as f:
        model = pickle.load(f)

    return data_columns, model


data_columns, model = load_artifacts()

# location columns start from index 3 (after sqft,bath,bhk)
location_list = data_columns[3:]

# ----------------- STREAMLIT UI -----------------
st.title("🏡 Bangalore Home Price Prediction App")
st.write("Enter the details below to predict the price of a house.")

sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=50)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)

location = st.selectbox("Select Location", location_list)

# ----------------- PRICE PREDICTION FUNCTION -----------------
def predict_price(location, sqft, bath, bhk):
    # Create zero array for all columns
    x = np.zeros(len(data_columns))

    # Fill basic features
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # One-hot encode location
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    # Predict using model
    return round(model.predict([x])[0], 2)


# ----------------- BUTTON -----------------
if st.button("Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"Estimated Price: **₹ {price} Lakhs**")
