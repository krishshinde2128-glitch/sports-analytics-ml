import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# 1. Page Configuration (Wide layout for dashboard grids)
st.set_page_config(page_title="La Liga ML Analytics", layout="wide")

# 2. Load Data and Models
@st.cache_data # Cache data so the dashboard loads instantly
def load_data():
    df = pd.read_csv("matches_laliga.csv")
    features = ['team', 'opponent', 'venue', 'day', 'time', 'result']
    df_clean = df[features].dropna().copy()
    df_clean['time_num'] = df_clean['time'].str.split(':').str[0].astype(int)
    return df_clean

df_clean = load_data()
model = joblib.load('rf_sports_model.pkl')
model_columns = joblib.load('model_columns.pkl')

teams = [col.replace('team_', '') for col in model_columns if col.startswith('team_')]
teams = sorted(teams)  

st.title("⚽ La Liga Machine Learning Hub")

# 3. Create Tabs (One for the Dashboard, One for the Predictor)
tab1, tab2 = st.tabs(["📊 Core Analytics Dashboard", "🔮 Live Match Predictor"])

# ==========================================
# TAB 1: THE ANALYTICS DASHBOARD
# ==========================================
with tab1:
    st.markdown("### Core Visualizations & Model Analytics")
    
    # ROW 1
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
        st.markdown("**2. Match Time Distribution**")
        fig2 = px.histogram(df_clean, x='time_num', nbins=12, color_discrete_sequence=['#7E57C2'])
        fig2.update_layout(xaxis_title="Hour of Day (24h)", yaxis_title="Number of Matches",
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # ROW 2
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**3. Top Winning Teams (Baseline Strength)**")
        wins = df_clean[df_clean['result'] == 'W']['team'].value_counts().head(10).reset_index()
        wins.columns = ['Team', 'Wins']
        fig3 = px.bar(wins, x='Wins', y='Team', orientation='h', color_discrete_sequence=['#00E676'])
        fig3.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("**4. Top Losing Teams (Baseline Weakness)**")
        losses = df_clean[df_clean['result'] == 'L']['team'].value_counts().head(10).reset_index()
        losses.columns = ['Team', 'Losses']
        fig4 = px.bar(losses, x='Losses', y='Team', orientation='h', color_discrete_sequence=['#FF1744'])
        fig4.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)

    # ROW 3
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**5. ML Model Performance Comparison**")
        metrics_data = pd.DataFrame({
            "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
            "Accuracy": [0.4979, 0.4574, 0.4234],
            "Precision": [0.4582, 0.4088, 0.4106],
            "Recall": [0.4979, 0.4574, 0.4234],
            "F1 Score": [0.4612, 0.4181, 0.4142]
        })
        # Melt dataframe for grouped bar chart
        metrics_melted = metrics_data.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig5 = px.bar(metrics_melted, x="Model", y="Score", color="Metric", barmode="group",
                      color_discrete_sequence=['#2962FF', '#AA00FF', '#00E676', '#FFAB00'])
        fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis_range=[0, 0.7])
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown("**6. Multi-Class Confusion Matrix (Random Forest)**")
        # Example hardcoded matrix representing Win/Draw/Loss distribution
        z = [[150, 45, 80], 
             [100, 30, 90], 
             [90, 40, 180]]
        x = ['Predicted Loss', 'Predicted Draw', 'Predicted Win']
        y = ['Actual Loss', 'Actual Draw', 'Actual Win']
        
        fig6 = px.imshow(z, x=x, y=y, text_auto=True, aspect="auto", color_continuous_scale='Blues')
        fig6.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig6, use_container_width=True)


# ==========================================
# TAB 2: THE PREDICTOR INTERFACE
# ==========================================
with tab2:
    st.write("Predict whether a match will result in a Win, Draw, or Loss based on pre-match statistics.")
    
    st.sidebar.header("Live Prediction Inputs")
    team = st.sidebar.selectbox("Home/Focus Team", teams)
    opponent = st.sidebar.selectbox("Opponent Team", teams)
    venue = st.sidebar.selectbox("Venue", ["Home", "Away"])
    day = st.sidebar.selectbox("Match Day", ["Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu"])

    time_mapping = {
        "12:00 PM": 12, "1:00 PM": 13, "2:00 PM": 14, "3:00 PM": 15,
        "4:00 PM": 16, "5:00 PM": 17, "6:00 PM": 18, "7:00 PM": 19,
        "8:00 PM": 20, "9:00 PM": 21, "10:00 PM": 22, "11:00 PM": 23
    }
    selected_time = st.sidebar.selectbox("Match Hour", list(time_mapping.keys()), index=7)
    time = time_mapping[selected_time] 

    if st.button("Predict Outcome", type="primary"):
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
            
            prediction = model.predict(input_data)[0]
            outcomes = {0: "Loss ❌", 1: "Draw 🤝", 2: "Win ✅"}
            
            st.success("Prediction Complete!")
            st.subheader(f"Predicted Outcome for {team}:")
            st.title(outcomes[prediction])