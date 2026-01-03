import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import io
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import CellIsRule

# --- PAGE CONFIG ---
st.set_page_config(page_title="NFL Betting Board", layout="wide")
st.title("🏈 NFL Live Betting Board")
st.markdown("""
**Instructions:**
1. Use the **"Matchup Settings"** table below to change QBs or update Vegas Lines.
2. The **"Live Projections"** table will instantly update with your new Model Lines and Bet Signals.
""")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    current_season = st.number_input("Season", value=2025)
    current_week = st.number_input("Week", value=19) # Playoffs
    
    st.divider()
    st.subheader("Decay Parameters")
    qb_decay = st.slider("QB Decay (Talent)", 0.90, 0.999, 0.99, help="Higher = History matters more")
    team_decay = st.slider("Team Decay (Recency)", 0.80, 0.99, 0.95, help="Lower = Only recent games matter")
    
    st.divider()
    st.subheader("Betting Rules")
    min_starts = st.number_input("Min Starts for Rating", value=3)
    edge_threshold = st.number_input("Std. Edge Required", value=2.0)
    key_val_threshold = st.number_input("Key Number Edge (3/7)", value=1.5)

# --- CACHED DATA LOADING ---
@st.cache_data
def load_data(seasons):
    schedule = nfl.load_schedules(seasons=seasons).to_pandas()
    stats = nfl.load_player_stats(seasons=seasons).to_pandas()
    if 'team' not in stats.columns: 
        stats['team'] = stats['recent_team']
    return schedule, stats

# --- MAIN APP LOGIC ---
try:
    # 1. LOAD & PREP
    schedule, stats = load_data([current_season-1, current_season])
    
    # Filter Valid Games for Training
    games = schedule[
        (schedule['game_type'].isin(['REG', 'POST'])) &
        (schedule['spread_line'].notnull()) &
        (schedule['result'].notnull())
    ].copy()
    
    # Get Current Slate
    current_slate = schedule[
        (schedule['season'] == current_season) & 
        (schedule['week'] == current_week)
    ].copy()

    # Normalize Signs (Negative = Favorite)
    games['market_line'] = -games['spread_line']
    current_slate['market_line'] = -current_slate['spread_line']

    # 2. MAP STARTERS
    passers = stats[stats['attempts'] > 0].sort_values('attempts', ascending=False)
    starters = passers.groupby(['season', 'week', 'team']).head(1)
    starter_map = {}
    for _, row in starters.iterrows():
        starter_map[(row['season'], row['week'], row['team'])] = row['player_display_name']

    def assign_starters(row):
        h = starter_map.get((row['season'], row['week'], row['home_team']))
        a = starter_map.get((row['season'], row['week'], row['away_team']))
        return pd.Series([h, a])

    games[['home_qb', 'away_qb']] = games.apply(assign_starters, axis=1)
    games = games.dropna(subset=['home_qb', 'away_qb']).copy()

    # Map Projected Starters for current week
    sorted_games = games.sort_values(['season', 'week'])
    last_qb_map = {}
    for _, row in sorted_games.iterrows():
        last_qb_map[row['home_team']] = row['home_qb']
        last_qb_map[row['away_team']] = row['away_qb']

    # --- REGRESSION MODELING ---
    
    # STAGE 1: QBs
    games['s1_weight'] = games.apply(lambda x: qb_decay ** ((current_season - x['season']) * 52 + (current_week - x['week'])), axis=1)
    h_team = pd.get_dummies(games['home_team'], dtype=int)
    a_team = pd.get_dummies(games['away_team'], dtype=int)
    h_qb = pd.get_dummies(games['home_qb'], dtype=int)
    a_qb = pd.get_dummies(games['away_qb'], dtype=int)
    
    all_teams = set(h_team.columns).union(a_team.columns)
    all_qbs = set(h_qb.columns).union(a_qb.columns)
    
    h_team = h_team.reindex(columns=all_teams, fill_value=0)
    a_team = a_team.reindex(columns=all_teams, fill_value=0)
    h_qb = h_qb.reindex(columns=all_qbs, fill_value=0)
    a_qb = a_qb.reindex(columns=all_qbs, fill_value=0)
    
    X = pd.concat([h_team.sub(a_team), h_qb.sub(a_qb)], axis=1)
    X['HFA'] = 1
    y = games['market_line']
    
    clf_qbs = Ridge(alpha=1.5, fit_intercept=False)
    clf_qbs.fit(X, y, sample_weight=games['s1_weight'])
    
    coefs = pd.Series(clf_qbs.coef_, index=X.columns)
    qb_ratings = coefs[list(all_qbs)]
    qb_ratings = qb_ratings - qb_ratings.mean()

    # Guardrails
    qb_counts = games['home_qb'].value_counts().add(games['away_qb'].value_counts(), fill_value=0)
    bad_qb_rating = qb_ratings.quantile(0.95) # High Positive = Bad (in this sign convention)
    for qb in qb_ratings.index:
        if qb_counts.get(qb, 0) < min_starts:
            if qb_ratings[qb] < bad_qb_rating:
                qb_ratings[qb] = bad_qb_rating
    
    qb_dict = qb_ratings.to_dict()

    # STAGE 2: Teams
    games_s2 = games[games['season'] == current_season].copy()
    def remove_qb_impact(row):
        h_val = qb_dict.get(row['home_qb'], bad_qb_rating) 
        a_val = qb_dict.get(row['away_qb'], bad_qb_rating)
        return row['market_line'] - (h_val - a_val)
    
    games_s2['roster_line'] = games_s2.apply(remove_qb_impact, axis=1)
    games_s2['s2_weight'] = games_s2.apply(lambda x: team_decay ** (current_week - x['week']), axis=1)
    
    h_team_s2 = pd.get_dummies(games_s2['home_team'], dtype=int)
    a_team_s2 = pd.get_dummies(games_s2['away_team'], dtype=int)
    h_team_s2 = h_team_s2.reindex(columns=all_teams, fill_value=0)
    a_team_s2 = a_team_s2.reindex(columns=all_teams, fill_value=0)
    
    X_s2 = h_team_s2.sub(a_team_s2)
    X_s2['HFA'] = 1
    y_s2 = games_s2['roster_line']
    
    clf_teams = Ridge(alpha=1.0, fit_intercept=False)
    clf_teams.fit(X_s2, y_s2, sample_weight=games_s2['s2_weight'])
    
    coefs_s2 = pd.Series(clf_teams.coef_, index=X_s2.columns)
    hfa_final = coefs_s2['HFA']
    team_ratings = coefs_s2.drop('HFA')
    team_dict = team_ratings.to_dict()

    # --- INTERACTIVE DASHBOARD LOGIC ---

    # 1. Prepare Inputs Table
    # We create a dataframe that holds the "State" of the QBs and Lines
    input_data = []
    
    for _, game in current_slate.iterrows():
        away_tm = game['away_team']
        home_tm = game['home_team']
        
        # Default QBs
        away_qb_def = last_qb_map.get(away_tm, "(Generic Backup)")
        home_qb_def = last_qb_map.get(home_tm, "(Generic Backup)")
        
        # Default Vegas
        vegas_def = game['market_line'] if pd.notnull(game['market_line']) else 0.0
        
        input_data.append({
            "Away Team": away_tm,
            "Away QB": away_qb_def,
            "Home Team": home_tm,
            "Home QB": home_qb_def,
            "Vegas (Home)": vegas_def
        })
        
    input_df = pd.DataFrame(input_data)
    
    # List of QBs for Dropdown (Sorted Alphabetically)
    qb_options = sorted(list(qb_ratings.index))
    qb_options.insert(0, "(Generic Backup)")

    # 2. RENDER INPUT TABLE (Data Editor)
    st.subheader("1. Matchup Settings")
    st.caption("Double-click any QB to change them. Edit the Vegas Line to match your sportsbook.")
    
    edited_df = st.data_editor(
        input_df,
        column_config={
            "Away QB": st.column_config.SelectboxColumn(
                "Away QB",
                help="Select the Away Starter",
                options=qb_options,
                required=True,
            ),
            "Home QB": st.column_config.SelectboxColumn(
                "Home QB",
                help="Select the Home Starter",
                options=qb_options,
                required=True,
            ),
            "Vegas (Home)": st.column_config.NumberColumn(
                "Vegas (Home Line)",
                help="Enter -7.0 for Home Favored by 7",
                format="%.1f"
            )
        },
        disabled=["Away Team", "Home Team"],
        hide_index=True,
        num_rows="fixed"
    )

    # 3. CALCULATE RESULTS LIVE
    results_data = []
    
    for _, row in edited_df.iterrows():
        # Get Edited Values
        a_tm = row['Away Team']
        h_tm = row['Home Team']
        a_qb = row['Away QB']
        h_qb = row['Home QB']
        vegas_line = row['Vegas (Home)']
        
        # Lookup Ratings
        # (Generic Backup) manual handling if it's not in dict
        r_a_tm = team_dict.get(a_tm, 0)
        r_h_tm = team_dict.get(h_tm, 0)
        r_a_qb = qb_dict.get(a_qb, bad_qb_rating)
        r_h_qb = qb_dict.get(h_qb, bad_qb_rating)
        
        # Calculate Model Line
        # Formula: (Home + HFA) - Away
        # Remember: Negative = Good. 
        model_line = (r_h_tm + r_h_qb + hfa_final) - (r_a_tm + r_a_qb)
        
        # Calculate Edge
        edge = model_line - vegas_line
        
        # Key Number Logic
        # Check if we crossed 3 or 7
        is_key = False
        if (model_line - 3) * (vegas_line - 3) < 0: is_key = True
        if (model_line - 7) * (vegas_line - 7) < 0: is_key = True
        if (model_line + 3) * (vegas_line + 3) < 0: is_key = True
        if (model_line + 7) * (vegas_line + 7) < 0: is_key = True
        
        req_edge = key_val_threshold if is_key else edge_threshold
        
        # Signal
        signal = "PASS"
        # Edge < -Threshold (Home is MORE negative/favored than Vegas) -> Bet Home
        if edge < -req_edge:
            signal = f"BET {h_tm}"
        # Edge > Threshold (Home is LESS favored/More positive) -> Bet Away
        elif edge > req_edge:
            signal = f"BET {a_tm}"
            
        results_data.append({
            "Matchup": f"{a_tm} @ {h_tm}",
            "Model Line": round(model_line, 1),
            "Vegas Line": vegas_line,
            "Edge": round(edge, 1),
            "Req. Edge": req_edge,
            "SIGNAL": signal
        })
        
    results_df = pd.DataFrame(results_data)

    # 4. RENDER RESULTS TABLE
    st.subheader("2. Live Projections")
    
    # Custom Styling for Signals
    def highlight_signal(s):
        return ['background-color: #d4edda; color: #155724; font-weight: bold' if 'BET' in v else '' for v in s]

    st.dataframe(
        results_df.style.apply(highlight_signal, subset=['SIGNAL']),
        hide_index=True,
        use_container_width=True
    )
    
    # 5. EXCEL DOWNLOAD (Optional Backup)
    st.divider()
    st.caption("Download the offline version for Excel:")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        edited_df.to_excel(writer, sheet_name='Settings', index=False)
        results_df.to_excel(writer, sheet_name='Projections', index=False)
        
    st.download_button(
        label="Download Results to Excel",
        data=buffer,
        file_name=f"NFL_Projections_Week_{current_week}.xlsx"
    )

except Exception as e:
    st.error(f"Error: {e}")
