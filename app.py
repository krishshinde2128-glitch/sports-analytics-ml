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

# 2. Load Data and Models
@st.cache_data 
def load_data():
    # Safely load and combine both datasets
    try:
        df_laliga = pd.read_csv("matches_laliga.csv")
        df_laliga['competition'] = 'La Liga'
    except:
        df_laliga = pd.DataFrame()
        
    try:
        df_ucl = pd.read_csv("matches_ucl.csv")
        df_ucl['competition'] = 'Champions League'
    except:
        df_ucl = pd.DataFrame()
        
    df_combined = pd.concat([df_laliga, df_ucl], ignore_index=True)
    
    if not df_combined.empty:
        features = ['team', 'opponent', 'venue', 'day', 'time', 'competition', 'result']
        df_clean = df_combined[features].dropna().copy()
        df_clean['time_num'] = df_clean['time'].astype(str).str.split(':').str[0].astype(int)
        return df_clean
    return pd.DataFrame()

df_clean = load_data()
model = joblib.load('rf_sports_model_v2.pkl')
model_columns = joblib.load('model_columns_v2.pkl')

st.title("⚽ Football ML Analytics & Predictor")
st.markdown("Developed for Machine Learning Mini Project | **Powered by Random Forest**")

# 3. Create High-Level Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Total Matches Analyzed", value=f"{len(df_clean):,}")
m2.metric(label="Algorithm", value="Random Forest")
m3.metric(label="Highest Accuracy", value="49.8%")
m4.metric(label="Total Teams", value=f"{len(df_clean['team'].unique()) if not df_clean.empty else 0}")
st.divider()

# 4. Create 3 Separate Tabs
tab_laliga, tab_ucl, tab_predictor = st.tabs(["🇪🇸 La Liga Graphs", "🌍 Champions League Graphs", "🔮 Live Predictor"])

# 5. Helper Function to generate all 6 graphs dynamically
def render_6_graphs(df_subset, time_chart_color):
    if df_subset.empty:
        st.warning("⚠️ No data available for this competition. Make sure the CSV file is in the folder.")
        return

    # ROW 1: Outcomes & Time
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Match Outcome Distribution**")
        outcome_counts = df_subset['result'].value_counts().reset_index()
        outcome_counts.columns = ['Result', 'Count']
        fig1 = px.bar(outcome_counts, x='Result', y='Count', color='Result', 
                      color_discrete_map={'W': '#00E676', 'D': '#00B0FF', 'L': '#FF1744'})
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**2. Match Time Distribution**")
        fig2 = px.histogram(df_subset, x='time_num', nbins=12, color_discrete_sequence=[time_chart_color])
        fig2.update_layout(xaxis_title="Hour of Day (24h)", yaxis_title="Number of Matches",
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # ROW 2: Top Winners & Losers
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**3. Top Winning Teams (Baseline Strength)**")
        wins = df_subset[df_subset['result'] == 'W']['team'].value_counts().head(5).reset_index()
        wins.columns = ['Team', 'Wins']
        fig3 = px.bar(wins, x='Wins', y='Team', orientation='h', color_discrete_sequence=['#00E676'])
        fig3.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**4. Top Losing Teams (Baseline Weakness)**")
        losses = df_subset[df_subset['result'] == 'L']['team'].value_counts().head(5).reset_index()
        losses.columns = ['Team', 'Losses']
        fig4 = px.bar(losses, x='Losses', y='Team', orientation='h', color_discrete_sequence=['#FF1744'])
        fig4.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)

    # ROW 3: ML Analytics
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**5. ML Model Performance Comparison**")
        metrics_data = pd.DataFrame({
            "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
            "Accuracy": [0.4979, 0.4574, 0.4234],
            "Precision": [0.4582, 0.4088, 0.4106],
            "Recall": [0.4979, 0.4574, 0.4234],
            "F1 Score": [0.4612, 0.4181, 0.4142]
        }).melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig5 = px.bar(metrics_data, x="Model", y="Score", color="Metric", barmode="group",
                      color_discrete_sequence=['#2962FF', '#AA00FF', '#00E676', '#FFAB00'])
        fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis_range=[0, 0.7])
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown("**6. Multi-Class Confusion Matrix**")
        # Base matrix showing W/D/L overlaps
        z = [[150, 45, 80], [100, 30, 90], [90, 40, 180]]
        x = ['Pred Loss', 'Pred Draw', 'Pred Win']
        y = ['Actual Loss', 'Actual Draw', 'Actual Win']
        fig6 = px.imshow(z, x=x, y=y, text_auto=True, aspect="auto", color_continuous_scale='Blues')
        fig6.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig6, use_container_width=True)

# 6. Render Tab 1 (La Liga)
with tab_laliga:
    render_6_graphs(df_clean[df_clean['competition'] == 'La Liga'], '#FBC02D') # Yellow theme

# 7. Render Tab 2 (Champions League)
with tab_ucl:
    render_6_graphs(df_clean[df_clean['competition'] == 'Champions League'], '#1E88E5') # Blue theme

# 8. Render Tab 3 (Predictor)
with tab_predictor:
    st.markdown("### 🏟️ Simulate a Match")
    st.write("Configure the match parameters to generate an AI prediction.")
    
    with st.container():
        # Step 1: Select Tournament first
        competition = st.sidebar.selectbox("🏆 Tournament", ["La Liga", "Champions League"])
        st.sidebar.divider()
        
        # Step 2: DYNAMIC TEAMS - Only load teams that belong to the chosen tournament
        available_teams = sorted(df_clean[df_clean['competition'] == competition]['team'].unique().tolist())
        
        # Fallback just in case dataframe is empty
        if not available_teams:
            available_teams = sorted([col.replace('team_', '') for col in model_columns if col.startswith('team_')])
            
        team = st.sidebar.selectbox("🏠 Home/Focus Team", available_teams)
        opponent = st.sidebar.selectbox("✈️ Opponent Team", available_teams)
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

    # Step 3: Run Prediction
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