import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Load Data from the 'Master Training' View
conn = sqlite3.connect('cricket_detailed.db')
query = """
SELECT 
    d.match_id,
    d.over,
    d.inning,
    m.venue,
    v.avg_runs_per_over AS venue_scoring_index,
    -- Ensure this column is selected!
    AVG(pm.strike_rate) AS matchup_strike_rate,
    t.total_over_runs AS actual_runs_in_over
FROM deliveries d
JOIN matches m ON d.match_id = m.match_id
LEFT JOIN view_venue_stats v ON m.venue = v.venue
LEFT JOIN view_player_matchups pm ON d.batter = pm.batter AND d.bowler = pm.bowler
JOIN (
    SELECT match_id, inning, over, SUM(runs_batter + runs_extras) as total_over_runs
    FROM deliveries
    GROUP BY match_id, inning, over
) t ON d.match_id = t.match_id AND d.inning = t.inning AND d.over = t.over
GROUP BY d.match_id, d.inning, d.over;
"""

df = pd.read_sql_query(query, conn)
conn.close()

# 2. Safety Check: If the join resulted in NO matches, the column might not exist.
# This prevents the KeyError 'matchup_strike_rate'
if 'matchup_strike_rate' not in df.columns:
    print("Column missing! Creating an empty matchup_strike_rate column.")
    df['matchup_strike_rate'] = 0.0

# 3. Fill the NaNs (for new players who don't have a history yet)
df['matchup_strike_rate'] = df['matchup_strike_rate'].fillna(df['matchup_strike_rate'].mean() if not df['matchup_strike_rate'].isna().all() else 100.0)

print("Data loaded successfully. Proceeding to training...")

# 2. Data Cleaning
# Fill missing matchup data with the average (for players who haven't met)
df['matchup_strike_rate'] = df['matchup_strike_rate'].fillna(df['matchup_strike_rate'].mean())

# 3. Encoding Categorical Data
# Convert Venue and Inning into numbers the AI can understand
df = pd.get_dummies(df, columns=['venue', 'inning'])

# 4. Define Features (X) and Target (y)
cols_to_drop = ['match_id', 'actual_runs_in_over']
X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
y = df['actual_runs_in_over']

# 5. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)
print(f"Mean Error: {mean_absolute_error(y_test, predictions):.2f} runs per over")
model.save_model("ipl_model_v1.json")
print("Model successfully saved to ipl_model_v1.json")