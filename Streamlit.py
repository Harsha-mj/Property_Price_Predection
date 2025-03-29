import streamlit as st
import numpy as np
import joblib

# Set the page title
st.title("🏡 Property Price Prediction App")
st.write("""
This app predicts the estimated price of a property 
based on various features such as size, age, and location.
""")

# Load the pre-trained model
model = joblib.load("property_price_model.pkl")

# Form for user input
with st.form("property_details"):
    st.subheader("Enter Property Details")

    # User inputs
    year_sold = st.number_input("Year Sold", min_value=1990, max_value=2025, value=2020)
    property_tax = st.number_input("Property Tax ($)", min_value=50, max_value=5000, value=500)
    insurance = st.number_input("Insurance Cost ($)", min_value=20, max_value=2000, value=200)
    beds = st.number_input("Number of Beds", min_value=1, max_value=10, value=3)
    baths = st.number_input("Number of Baths", min_value=1, max_value=10, value=2)
    sqft = st.number_input("Size in Sqft", min_value=300, max_value=10000, value=2000)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    lot_size = st.number_input("Lot Size", min_value=0, max_value=500000, value=5000)
    basement = st.selectbox("Basement Available?", [0, 1])
    bunglow = st.selectbox("Is it a Bungalow?", [0, 1])
    condo = st.selectbox("Is it a Condo?", [0, 1])
    popular = st.selectbox("Is the area popular?", [0, 1])
    property_age = st.number_input("Property Age", min_value=0, max_value=200, value=10)

    # Submit button
    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Prepare input data
    input_data = np.array([
        year_sold, property_tax, insurance, beds, baths, sqft,
        year_built, lot_size, basement, bunglow, condo, popular, property_age
    ]).reshape(1, -1)

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    # Display prediction
    st.subheader("🏠 Estimated Property Price:")
    st.success(f"💲 {predicted_price:,.2f}")

# Feature importance (optional)
st.write("Feature Importance of the model:")
st.image("feature_importance.png", caption="Feature importance visualization", use_column_width=True)
