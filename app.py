%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and expected columns
model = joblib.load('rf_sports_model.pkl')
model_columns = joblib.load('model_columns.pkl')

teams = [col.replace('team_', '') for col in model_columns if col.startswith('team_')]
teams = sorted(teams)  

st.title("⚽ La Liga Match Outcome Predictor")
st.write("Predict whether a match will result in a Win, Draw, or Loss based on pre-match statistics.")

st.sidebar.header("Match Details")
team = st.sidebar.selectbox("Home/Focus Team", teams)
opponent = st.sidebar.selectbox("Opponent Team", teams)
venue = st.sidebar.selectbox("Venue", ["Home", "Away"])
day = st.sidebar.selectbox("Match Day", ["Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu"])
time = st.sidebar.slider("Match Hour (24h format)", 12, 23, 19)

if st.button("Predict Outcome"):
    if team == opponent:
        st.warning("⚠️ Team and Opponent cannot be the same!")
    else:
        input_data = pd.DataFrame(columns=model_columns)
        input_data.loc[0] = 0  
        
        input_data.loc[0, 'time'] = time
        
        if f'team_{team}' in model_columns: 
            input_data.loc[0, f'team_{team}'] = 1
        if f'opponent_{opponent}' in model_columns: 
            input_data.loc[0, f'opponent_{opponent}'] = 1
        if f'venue_{venue}' in model_columns: 
            input_data.loc[0, f'venue_{venue}'] = 1
        if f'day_{day}' in model_columns: 
            input_data.loc[0, f'day_{day}'] = 1
        
        prediction = model.predict(input_data)[0]
        outcomes = {0: "Loss ❌", 1: "Draw 🤝", 2: "Win ✅"}
        
        st.subheader(f"Predicted Outcome for {team}:")
        st.title(outcomes[prediction])