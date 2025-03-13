import streamlit as st
import numpy as np
import joblib  # To load the model
import pandas as pd

# Load the trained model and scaler
model = joblib.load("./saved_models/Sales_Prediction_Model.pkl")
scaler = joblib.load("./saved_models/scalers.pkl")

# Title
st.title("Sales Prediction Web App")
st.markdown("### Enter the product and outlet details below to predict sales")

# --- User Input Fields ---

# Numeric Inputs
weight = st.number_input("Weight (in kg)", min_value=0.0, max_value=50.0, step=0.1)
product_visibility = st.number_input(
    "Product Visibility", min_value=0.0, max_value=1.0, step=0.01
)
product_type = st.selectbox(
    "Product Type",
    [
        "Dairy",
        "Soft Drinks",
        "Meat",
        "Fruits & Vegetables",
        "Household",
        "Baking Goods",
        "Snack Foods",
        "Frozen Foods",
        "Breakfast",
        "Health & Hygiene",
        "Hard Drinks",
        "Canned",
        "Breads",
        "Starchy Foods",
        "Others",
    ],
)

mrp = st.number_input("MRP (in ₹)", min_value=0.0, max_value=300.0, step=0.1)
outlet_age = st.number_input("Outlet Age (in years)", min_value=0, max_value=50)

# Categorical Inputs (Converted to Numeric)
fat_content = st.selectbox("Fat Content", ["Low Fat", "Regular"])
outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
location_type = st.selectbox("Location Type", ["Tier 1", "Tier 2", "Tier 3"])
outlet_type = st.selectbox(
    "Outlet Type",
    ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"],
)

# --- Convert Categorical Values to Numbers ---
fat_content_map = {"Low Fat": 0, "Regular": 1}
outlet_size_map = {"Small": 0, "Medium": 1, "High": 2}
location_type_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
outlet_type_map = {
    "Grocery Store": 0,
    "Supermarket Type1": 1,
    "Supermarket Type2": 2,
    "Supermarket Type3": 3,
}
product_type_map = {  # Example mapping
    "Dairy": 0,
    "Soft Drinks": 1,
    "Meat": 2,
    "Fruits & Vegetables": 3,
    "Household": 4,
    "Baking Goods": 5,
    "Snack Foods": 6,
    "Frozen Foods": 7,
    "Breakfast": 8,
    "Health & Hygiene": 9,
    "Hard Drinks": 10,
    "Canned": 11,
    "Breads": 12,
    "Starchy Foods": 13,
    "Others": 14,
}

fat_content = fat_content_map[fat_content]
outlet_size = outlet_size_map[outlet_size]
location_type = location_type_map[location_type]
outlet_type = outlet_type_map[outlet_type]
product_type = product_type_map[product_type]

# --- Predict Sales ---
if st.button("Predict Sales"):
    # Create input array
    input_data = np.array(
        [
            [
                weight,
                fat_content,
                product_visibility,
                product_type,
                mrp,
                outlet_size,
                location_type,
                outlet_type,
                outlet_age,
            ]
        ]
    )

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict sales
    prediction = model.predict(input_data_scaled)[0]

    # Display result
    st.success(f"Predicted Sales: ₹{prediction:.2f}")
