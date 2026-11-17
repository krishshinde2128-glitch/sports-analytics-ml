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

# --- NEW SECTION: Dataset Info & Model Comparison ---
with st.expander("📊 View Dataset & Model Performance Details"):
    st.write("**Dataset Source:** matches_laliga.csv")
    st.write("**Total Records Analyzed:** 4,700 historical matches")
    
    st.write("### Model Comparison")
    st.write("Below is the performance comparison of the classification models tested during development:")
    
    # Exact metrics from your dataset run
    metrics_data = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.4979, 0.4574, 0.4234],
        "Precision": [0.4582, 0.4088, 0.4106],
        "Recall": [0.4979, 0.4574, 0.4234],
        "F1-Score": [0.4612, 0.4181, 0.4142]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    # Display as a static table just like the project requirements PDF
    st.table(metrics_df.set_index("Model"))
    
    st.caption("*Note: Predicting sports matches relies heavily on unpredictable human factors. The models perform better than random guessing (33%), highlighting the baseline difficulty of sports analytics.*")
# ----------------------------------------------------

st.sidebar.header("Match Details")
team = st.sidebar.selectbox("Home/Focus Team", teams)
opponent = st.sidebar.selectbox("Opponent Team", teams)
venue = st.sidebar.selectbox("Venue", ["Home", "Away"])
day = st.sidebar.selectbox("Match Day", ["Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu"])

# Mapping for 12-hour display to 24-hour model input
time_mapping = {
    "12:00 PM": 12, "1:00 PM": 13, "2:00 PM": 14, "3:00 PM": 15,
    "4:00 PM": 16, "5:00 PM": 17, "6:00 PM": 18, "7:00 PM": 19,
    "8:00 PM": 20, "9:00 PM": 21, "10:00 PM": 22, "11:00 PM": 23
}

selected_time = st.sidebar.selectbox("Match Hour", list(time_mapping.keys()), index=7)
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