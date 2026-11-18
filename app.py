import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Football ML Analytics", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    .stProgress > div > div > div > div { background-color: #00E676; }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Data and Models (Updated to v2 files!)
@st.cache_data 
def load_data():
    df_laliga = pd.read_csv("matches_laliga.csv")
    df_ucl = pd.read_csv("matches_ucl.csv")
    df_laliga['competition'] = 'La Liga'
    df_ucl['competition'] = 'Champions League'
    df_combined = pd.concat([df_laliga, df_ucl], ignore_index=True)
    
    features = ['team', 'opponent', 'venue', 'day', 'time', 'competition', 'result']
    df_clean = df_combined[features].dropna().copy()
    df_clean['time_num'] = df_clean['time'].astype(str).str.split(':').str[0].astype(int)
    return df_clean

df_clean = load_data()

# Load the NEW models you just trained
model = joblib.load('rf_sports_model_v2.pkl')
model_columns = joblib.load('model_columns_v2.pkl')

teams = [col.replace('team_', '') for col in model_columns if col.startswith('team_')]
teams = sorted(teams)  

st.title("⚽ Football ML Analytics & Predictor")
st.markdown("Developed for Machine Learning Mini Project | **Powered by Random Forest**")

# 3. Create Tabs
tab1, tab2 = st.tabs(["📊 Core Analytics Dashboard", "🔮 Live Match Predictor"])

# ==========================================
# TAB 1: THE ANALYTICS DASHBOARD
# ==========================================
with tab1:
    st.markdown("### 🏆 Project Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Total Matches Analyzed", value=f"{len(df_clean):,}")
    m2.metric(label="Algorithm", value="Random Forest")
    m3.metric(label="Leagues", value="La Liga + UCL")
    m4.metric(label="Total Teams", value=f"{len(teams)}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Match Outcome Distribution**")
        outcome_counts = df_clean['result'].value_counts().reset_index()
        outcome_counts.columns = ['Result', 'Count']
        fig1 = px.bar(outcome_counts, x='Result', y='Count', color='Result', 
                      color_discrete_map={'W': '#00E676', 'D': '#00B0FF', 'L': '#FF1744'})
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**2. Top Winning Teams (Baseline Strength)**")
        wins = df_clean[df_clean['result'] == 'W']['team'].value_counts().head(5).reset_index()
        wins.columns = ['Team', 'Wins']
        fig2 = px.bar(wins, x='Wins', y='Team', orientation='h', color_discrete_sequence=['#00E676'])
        fig2.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# TAB 2: THE PREDICTOR INTERFACE
# ==========================================
with tab2:
    st.markdown("### 🏟️ Simulate a Match")
    st.write("Configure the match parameters on the sidebar to generate an AI prediction.")
    
    with st.container():
        st.sidebar.header("Live Prediction Inputs")
        
        # --- FIXED: Competition dropdown placed correctly in the sidebar ---
        competition = st.sidebar.selectbox("🏆 Tournament", ["La Liga", "Champions League"])
        st.sidebar.divider()
        
        team = st.sidebar.selectbox("🏠 Home/Focus Team", teams)
        opponent = st.sidebar.selectbox("✈️ Opponent Team", teams)
        venue = st.sidebar.selectbox("📍 Venue", ["Home", "Away"])
        
        st.sidebar.divider()
        
        day = st.sidebar.selectbox("📅 Match Day", ["Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu"])
        time_mapping = {
            "12:00 PM": 12, "1:00 PM": 13, "2:00 PM": 14, "3:00 PM": 15,
            "4:00 PM": 16, "5:00 PM": 17, "6:00 PM": 18, "7:00 PM": 19,
            "8:00 PM": 20, "9:00 PM": 21, "10:00 PM": 22, "11:00 PM": 23
        }
        selected_time = st.sidebar.selectbox("⏰ Match Hour", list(time_mapping.keys()), index=7)
        time = time_mapping[selected_time] 

        st.sidebar.divider()

    if st.sidebar.button("🔮 Predict Match Outcome", use_container_width=True, type="primary"):
        if team == opponent:
            st.error("⚠️ Team and Opponent cannot be the same!")
        else:
            input_data = pd.DataFrame(columns=model_columns)
            input_data.loc[0] = 0  
            input_data.loc[0, 'time'] = time
            if f'team_{team}' in model_columns: input_data.loc[0, f'team_{team}'] = 1
            if f'opponent_{opponent}' in model_columns: input_data.loc[0, f'opponent_{opponent}'] = 1
            if f'venue_{venue}' in model_columns: input_data.loc[0, f'venue_{venue}'] = 1
            if f'day_{day}' in model_columns: input_data.loc[0, f'day_{day}'] = 1
            
            # --- FIXED: Competition logic safely inside the button block ---
            if f'competition_{competition}' in model_columns: 
                input_data.loc[0, f'competition_{competition}'] = 1
            
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = max(probabilities) * 100
            
            outcomes = {0: "Loss ❌", 1: "Draw 🤝", 2: "Win ✅"}
            
            st.success("✅ Analysis Complete!")
            st.metric(label=f"Predicted Outcome for {team}:", value=outcomes[prediction])
            st.write(f"**Model Confidence:** {confidence:.1f}%")
            st.progress(int(confidence))
            
            if prediction == 2:
                st.balloons()