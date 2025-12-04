
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the Saved Artifacts artifacts
@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('ohe.pkl', 'rb') as f:
        ohe = pickle.load(f)
    with open('locality_means.pkl', 'rb') as f:
        locality_means = pickle.load(f)
    with open('global_mean.pkl', 'rb') as f:
        global_mean = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    return model, ohe, locality_means, global_mean, model_columns

model, ohe, locality_means, global_mean, model_columns = load_artifacts()

# Build the UI 
st.title("🏡 Real Estate Rent Predictor [2024 Dataset Model Trained]")
st.markdown("Enter the details of the property to get an estimated monthly rent.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    # City Selection
    # Extract cities from the OneHotEncoder categories
    cities = [x.split('_')[1] for x in ohe.get_feature_names_out(['city'])]
    city = st.selectbox("City", cities)
    
    # Locality Selection
    # We use the keys from our locality_means dictionary
    localities = sorted(list(locality_means.keys()))
    locality = st.selectbox("Locality", localities)

    # Numerical Inputs
    area = st.number_input("Area (Sq. Ft.)", min_value=100, max_value=10000, value=850)
    beds = st.number_input("Bedrooms (BHK)", min_value=1, max_value=10, value=2)

with col2:
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    balconies = st.number_input("Balconies", min_value=0, max_value=5, value=1)
    
    # Furnishing Mapping
    furnishing_options = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
    furnishing = st.selectbox("Furnishing Status", list(furnishing_options.keys()))

#  Prediction Logic 
if st.button("Predict Rent"):
    try:
        # 1. Prepare raw input dataframe
        input_data = pd.DataFrame({
            'city': [city],
            'locality': [locality],
            'area': [area],
            'beds': [beds],
            'bathrooms': [bathrooms],
            'balconies': [balconies],
            'furnishing_score': [furnishing_options[furnishing]]
        })

        # 2. Preprocessing (Must match notebook exactly!)
        
        # A. One-Hot Encoding for City
        city_encoded = ohe.transform(input_data[["city"]]).toarray()
        city_df = pd.DataFrame(city_encoded, columns=ohe.get_feature_names_out(['city']))
        
        # B. Concatenate and drop original city column
        input_data = pd.concat([input_data.drop('city', axis=1), city_df], axis=1)

        # C. Target Encoding for Locality
        # Map the locality to its mean rent. If new/unknown, use global_mean
        input_data['locality_encoded'] = input_data['locality'].map(locality_means)
        input_data['locality_encoded'] = input_data['locality_encoded'].fillna(global_mean)
        input_data.drop('locality', axis=1, inplace=True)

        # D. Ensure Column Order matches the Model's expectation
        # This handles any column reordering that happened during training
        input_data = input_data[model_columns]

        # 3. Make Prediction
        prediction = model.predict(input_data)[0]

        # 4. Display Result
        st.success(f"💰 Estimated Monthly Rent: ₹ {int(prediction):,}")
        
    except Exception as e:

        st.error(f"An error occurred: {e}")


