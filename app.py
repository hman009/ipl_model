import streamlit as st
import sqlite3
import pandas as pd
import xgboost as xgb

# --- 1. Load the Model ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("ipl_model_v1.json")
    return model

# --- 2. Database Helper Functions ---
def get_db_connection():
    return sqlite3.connect('cricket_detailed.db')

def get_options():
    conn = get_db_connection()
    venues = pd.read_sql("SELECT DISTINCT venue FROM matches ORDER BY venue", conn)['venue'].tolist()
    batters = pd.read_sql("SELECT DISTINCT batter FROM deliveries ORDER BY batter", conn)['batter'].tolist()
    bowlers = pd.read_sql("SELECT DISTINCT bowler FROM deliveries ORDER BY bowler", conn)['bowler'].tolist()
    conn.close()
    return venues, batters, bowlers

def fetch_stats(venue, batter, bowler):
    conn = get_db_connection()
    
    # 1. Get Venue Index (The "Ground Factor")
    v_query = "SELECT avg_runs_per_over FROM view_venue_stats WHERE venue = ?"
    v_res = pd.read_sql(v_query, conn, params=(venue,))
    # If the venue is new, we use 7.5 as a safe average
    v_idx = v_res['avg_runs_per_over'].iloc[0] if not v_res.empty else 7.5
    
    # 2. Get Matchup SR (The "Player Factor")
    m_query = "SELECT strike_rate FROM view_player_matchups WHERE batter = ? AND bowler = ?"
    m_res = pd.read_sql(m_query, conn, params=(batter, bowler))
    
    if not m_res.empty:
        # We have specific history for these two players
        m_sr = m_res['strike_rate'].iloc[0]
    else:
        # FALLBACK: If they've never faced each other, use the batter's career SR
        b_query = "SELECT AVG(strike_rate) as career_sr FROM view_player_matchups WHERE batter = ?"
        b_res = pd.read_sql(b_query, conn, params=(batter,))
        
        # If the batter has zero history (rare), use 100.0
        if not b_res.empty and pd.notnull(b_res['career_sr'].iloc[0]):
            m_sr = b_res['career_sr'].iloc[0]
        else:
            m_sr = 100.0
    
    conn.close()
    return v_idx, m_sr
# --- 3. UI Layout ---
st.set_page_config(page_title="IPL Predictor", page_icon="🏏")
st.title("🏏 IPL Over-by-Over Predictor")
st.write("Predict runs based on specific matchups and venue conditions.")

model = load_model()
venues, batters, bowlers = get_options()

col1, col2 = st.columns(2)

with col1:
    selected_venue = st.selectbox("Select Venue", venues, index=0)
    selected_batter = st.selectbox("Striker (Main)", batters, index=0)
    selected_bowler = st.selectbox("Bowler", bowlers, index=0)

with col2:
    over_num = st.slider("Over Number", 0, 19, 15)
    inning = st.radio("Inning", [1, 2], horizontal=True)

# --- 4. Prediction Logic ---
if st.button("Predict Runs", type="primary"):
    # 1. Fetch hidden stats from DB
    v_idx, m_sr = fetch_stats(selected_venue, selected_batter, selected_bowler)
    
    # 2. Get the exact feature names the model expects (44 columns)
    expected_features = model.get_booster().feature_names
    
    # 3. Create a DataFrame filled with 0s
    input_df = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # 4. Fill the base numerical features
    input_df['over'] = over_num
    input_df['venue_scoring_index'] = v_idx
    input_df['matchup_strike_rate'] = m_sr
    
    # 5. Handle the One-Hot Encoded columns (Venue and Inning)
    # We set the specific selected venue and inning columns to 1
    venue_col = f"venue_{selected_venue}"
    inning_col = f"inning_{inning}"
    
    if venue_col in input_df.columns:
        input_df[venue_col] = 1
    else:
        # Fallback if a venue name has special characters or mismatch
        st.warning(f"Note: Specific patterns for '{selected_venue}' might be missing from training.")

    if inning_col in input_df.columns:
        input_df[inning_col] = 1

    # 6. Predict
    try:
        prediction = model.predict(input_df)[0]
        
        # 7. Display Results
        st.divider()
        st.metric(label="Predicted Runs in Over", value=f"{max(0, round(prediction, 2))}")
        
        with st.expander("View AI Reasoning"):
            st.write(f"**Venue Advantage:** {v_idx:.2f} avg runs/over")
            st.write(f"**Historical Matchup SR:** {m_sr:.2f}")
            st.write(f"**Game Stage:** Over {over_num}, Inning {inning}")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")