import sqlite3
import json
import glob
import os

# 2. The Loading Function
def load_match_to_db(data, cursor):
    info = data['info']
    match_id = info['event'].get('match_number', 0)
    
    # Insert Match Info
    cursor.execute('INSERT OR IGNORE INTO matches VALUES (?,?,?)', 
                   (match_id, info.get('venue'), info['outcome'].get('winner')))
    
    # Insert Ball-by-Ball
    for i, inning in enumerate(data['innings']):
        for over_data in inning['overs']:
            over_num = over_data['over']
            for b_idx, d in enumerate(over_data['deliveries']):
                cursor.execute('''
                    INSERT OR IGNORE INTO deliveries (match_id, inning, over, ball, batter, bowler, runs_batter, runs_extras, is_wicket)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (match_id, i+1, over_num, b_idx+1, d['batter'], d['bowler'], 
                      d['runs']['batter'], d['runs']['extras'], 1 if 'wicket' in d else 0))

def build_database(folder_path, db_name='cricket_detailed.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY,
            venue TEXT,
            winner TEXT
        );

        CREATE TABLE IF NOT EXISTS deliveries (
            match_id INTEGER,
            inning INTEGER,
            over INTEGER,
            ball INTEGER,
            batter TEXT,
            bowler TEXT,
            runs_batter INTEGER,
            runs_extras INTEGER,
            is_wicket INTEGER,
            -- This replaces delivery_id and prevents the 776 runs bug:
            PRIMARY KEY (match_id, inning, over, ball),
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );
    ''')

    # Get a list of all JSON files in the folder
    files = glob.glob(os.path.join(folder_path, "*.json"))
    print(f"Found {len(files)} matches to process.")

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Use a 'try-except' or check if 'info' exists to avoid crashes
            if 'info' in data:
                # Call your loading logic here
                load_match_to_db(data, cursor)
                print(f"Successfully loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    conn.commit()
    conn.close()
    print("Database build complete!")

fold_path = "/home/divkrash/Desktop/nonagon_labs/ml_stuff/ipl_model/ipl_json"
build_database(fold_path)            

def create_view():
    conn = sqlite3.connect('cricket_detailed.db')
    cursor = conn.cursor()

    # Create the Matchup View
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS view_player_matchups AS
    SELECT 
        batter, 
        bowler,
        COUNT(*) AS balls_faced,
        SUM(runs_batter) AS total_runs,
        SUM(is_wicket) AS dismissals,
        (CAST(SUM(runs_batter) AS FLOAT) / COUNT(*)) * 100 AS strike_rate
    FROM deliveries
    GROUP BY batter, bowler;
    ''')
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS view_venue_stats AS
    SELECT 
        m.venue,
        AVG(d.runs_batter + d.runs_extras) * 6 AS avg_runs_per_over,
        AVG(d.is_wicket) * 6 AS avg_wickets_per_over
    FROM deliveries d
    JOIN matches m ON d.match_id = m.match_id
    GROUP BY m.venue;
    ''')  
    # cursor.execute("DROP VIEW IF EXISTS view_training_data") 
    cursor.execute('''
    CREATE VIEW view_training_data AS
SELECT 
    d.match_id,
    d.over,
    d.inning,
    m.venue,
    v.avg_runs_per_over AS venue_scoring_index,
    AVG(pm.strike_rate) AS matchup_strike_rate,
    t.total_over_runs AS actual_runs_in_over
FROM deliveries d
JOIN matches m ON d.match_id = m.match_id
LEFT JOIN view_venue_stats v ON m.venue = v.venue
LEFT JOIN view_player_matchups pm ON d.batter = pm.batter AND d.bowler = pm.bowler
JOIN (
    -- This calculates the actual total per over first
    SELECT match_id, inning, over, SUM(runs_batter + runs_extras) as total_over_runs
    FROM deliveries
    GROUP BY match_id, inning, over
) t ON d.match_id = t.match_id AND d.inning = t.inning AND d.over = t.over
GROUP BY d.match_id, d.inning, d.over;
''')

    conn.commit()
    print(" views created successfully.")

create_view()    