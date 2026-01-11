import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="pal14/visitWithUs", filename="best_classification_model_visitWithUs.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction")
st.write("""
This application predicts predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
Please enter the app details below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=0.0, max_value=100.0, step=1)
TypeofContact = st.selectbox("Contact Type", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])

Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=0, max_value=100000000)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3., 4., 5.])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0.0, max_value=10000.0)
Passport = st.selectbox("Passport", [0, 1])

OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.selectbox("Number of children visiting", [0., 1., 2., 3.])
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Long Ads per Hour", min_value=0.0, max_value=1000000.)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Package Taken: {prediction}")
