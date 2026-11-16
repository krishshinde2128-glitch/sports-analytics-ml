import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and expected columns
model = joblib.load('rf_sports_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("⚽ La Liga Match Outcome Predictor")
st.write("Predict whether a match will result in a Win, Draw, or Loss based on pre-match statistics.")

# Sidebar inputs
st.sidebar.header("Match Details")
team = st.sidebar.text_input("Home/Focus Team (e.g., Barcelona)")
opponent = st.sidebar.text_input("Opponent Team (e.g., Real Madrid)")
venue = st.sidebar.selectbox("Venue", ["Home", "Away"])
day = st.sidebar.selectbox("Match Day", ["Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu"])
time = st.sidebar.slider("Match Hour (24h format)", 12, 23, 19)

if st.button("Predict Outcome"):
    # Create an empty dataframe with the exact columns used during training
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0  # Fill initially with zeros
    
    # Update numerical columns
    input_data['time'] = time
    
    # Update categorical columns (set the specific 1-hot encoded column to True/1)
    if f'team_{team}' in model_columns: input_data[f'team_{team}'] = 1
    if f'opponent_{opponent}' in model_columns: input_data[f'opponent_{opponent}'] = 1
    if f'venue_{venue}' in model_columns: input_data[f'venue_{venue}'] = 1
    if f'day_{day}' in model_columns: input_data[f'day_{day}'] = 1
    
    # Make Prediction
    prediction = model.predict(input_data)[0]
    outcomes = {0: "Loss ❌", 1: "Draw 🤝", 2: "Win ✅"}
    
    st.subheader(f"Predicted Outcome for {team}:")
    st.title(outcomes[prediction])
