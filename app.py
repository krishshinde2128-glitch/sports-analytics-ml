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

# Create a mapping for 12-hour display to 24-hour model input
time_mapping = {
    "12:00 PM": 12, "1:00 PM": 13, "2:00 PM": 14, "3:00 PM": 15,
    "4:00 PM": 16, "5:00 PM": 17, "6:00 PM": 18, "7:00 PM": 19,
    "8:00 PM": 20, "9:00 PM": 21, "10:00 PM": 22, "11:00 PM": 23
}

# Display 12-hour options to the user, defaulting to 7:00 PM (index 7)
selected_time = st.sidebar.selectbox("Match Hour", list(time_mapping.keys()), index=7)
# Convert the user's selection back to the 24-hour integer for the model
time = time_mapping[selected_time] 

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