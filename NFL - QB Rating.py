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
st.set_page_config(page_title="NFL Betting Model", layout="wide")
st.title("🏈 NFL Market-Implied Power Ratings")
st.markdown("""
This tool reverse-engineers the betting market to separate **QB Talent** from **Team Power**.
It generates an **Interactive Excel File** where you can toggle QBs to see how the spread changes.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Model Settings")
current_season = st.sidebar.number_input("Season", value=2025)
current_week = st.sidebar.number_input("Current Week", value=19) # Playoffs!

st.sidebar.subheader("Decay Rates")
qb_decay = st.sidebar.slider("QB Talent Decay (Long Term)", 0.90, 0.999, 0.99)
team_decay = st.sidebar.slider("Team Power Decay (Short Term)", 0.80, 0.99, 0.95)

st.sidebar.subheader("Guardrails")
min_starts = st.sidebar.number_input("Min Starts for Rating", value=3)
edge_threshold = st.sidebar.number_input("Edge Threshold (Points)", value=1.5)

# --- CACHED DATA LOADING ---
@st.cache_data
def load_and_prep_data(seasons, target_week):
    with st.spinner("Fetching NFL Data..."):
        schedule = nfl.load_schedules(seasons=seasons).to_pandas()
        stats = nfl.load_player_stats(seasons=seasons).to_pandas()
        if 'team' not in stats.columns: 
            stats['team'] = stats['recent_team']
        return schedule, stats

# --- MAIN LOGIC ---
try:
    # 1. Load Data
    schedule, stats = load_and_prep_data([current_season-1, current_season], current_week)
    
    # Filter for Completed Games (Training Data)
    games = schedule[
        (schedule['game_type'].isin(['REG', 'POST'])) &
        (schedule['spread_line'].notnull()) &
        (schedule['result'].notnull())
    ].copy()
    
    # Filter for Current Slate (Prediction Data)
    current_slate = schedule[
        (schedule['season'] == current_season) & 
        (schedule['week'] == current_week)
    ].copy()

    # FIX SIGNS: Ensure Negative = Favorite
    games['market_line'] = -games['spread_line']
    current_slate['market_line'] = -current_slate['spread_line']

    # 2. Map Starters
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
    
    # Map Projected Starters (Most Recent)
    sorted_games = games.sort_values(['season', 'week'])
    last_qb_map = {}
    for _, row in sorted_games.iterrows():
        last_qb_map[row['home_team']] = row['home_qb']
        last_qb_map[row['away_team']] = row['away_qb']

    # --- STAGE 1: QB REGRESSION ---
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

    # --- GUARDRAILS ---
    qb_counts = games['home_qb'].value_counts().add(games['away_qb'].value_counts(), fill_value=0)
    # 95th Percentile because Good QBs are Negative (e.g., -6.0). Bad QBs are Positive (e.g., +6.0).
    bad_qb_rating = qb_ratings.quantile(0.95) 

    for qb in qb_ratings.index:
        if qb_counts.get(qb, 0) < min_starts:
            # If they are BETTER (lower) than bad, force them to be BAD.
            if qb_ratings[qb] < bad_qb_rating:
                qb_ratings[qb] = bad_qb_rating
    
    qb_dict = qb_ratings.to_dict()

    # --- STAGE 2: TEAM REGRESSION ---
    games_s2 = games[games['season'] == current_season].copy()
    
    def remove_qb_impact(row):
        h_val = qb_dict.get(row['home_qb'], bad_qb_rating) 
        a_val = qb_dict.get(row['away_qb'], bad_qb_rating)
        return row['market_line'] - (h_val - a_val)

    games_s2['roster_line'] = games_s2.apply(remove_qb_impact, axis=1)
    games_s2['s2_weight'] = games_s2.apply(lambda x: team_decay ** (current_week - x['week']), axis=1)

    h_team_s2 = pd.get_dummies(games_s2['home_team'], dtype=int)
    a_team_s2 = pd.get_dummies(games_s2['away_team'], dtype=int)
    
    # Reindex using ALL teams from Stage 1 to ensure consistency
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

    # --- PREPARE EXCEL DOWNLOAD ---
    st.success(f"Model Trained Successfully! HFA: {hfa_final:.2f}")

    # Prepare Dataframes
    df_qbs = pd.DataFrame({'QB': qb_ratings.index, 'Rating': qb_ratings.values}).sort_values('Rating', ascending=True) # Ascending = Best First (Negative)
    df_teams = pd.DataFrame({'Team': team_ratings.index, 'Rating': team_ratings.values}).sort_values('Rating', ascending=True)
    
    # Add Generic Backup
    backup_row = pd.DataFrame({'QB': ['(Generic Backup)'], 'Rating': [bad_qb_rating]})
    df_qbs = pd.concat([backup_row, df_qbs], ignore_index=True)

    # 1. Create In-Memory Excel File (Pandas Step)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_teams.to_excel(writer, sheet_name='DB_Teams', index=False)
        df_qbs.to_excel(writer, sheet_name='DB_QBs', index=False)
        pd.DataFrame({'Metric': ['HFA', 'Threshold'], 'Value': [hfa_final, edge_threshold]}).to_excel(writer, sheet_name='Settings', index=False)
        writer.book.create_sheet('Betting Board')
    
    # 2. Open In-Memory File to Add Formatting (OpenPyXL Step)
    buffer.seek(0)
    wb = load_workbook(buffer)
    ws = wb['Betting Board']
    
    # --- BUILD DASHBOARD ---
    headers = ["Away Team", "Away QB", "Home Team", "Home QB", "Model (HOME)", "Vegas (HOME)", "Edge", "SIGNAL"]
    ws.append(headers)

    # Styles
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    center = Alignment(horizontal="center")
    input_fill = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")

    for col_num, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center

    # Dropdowns
    qb_dv = DataValidation(type="list", formula1="='DB_QBs'!$A$2:$A$300", allow_blank=True)
    ws.add_data_validation(qb_dv)

    row_idx = 2
    for _, game in current_slate.iterrows():
        away_tm = game['away_team']
        home_tm = game['home_team']
        
        vegas_val = game['market_line'] 
        if pd.isna(vegas_val): vegas_val = "N/A"
        
        away_qb = last_qb_map.get(away_tm, "(Generic Backup)")
        home_qb = last_qb_map.get(home_tm, "(Generic Backup)")
        
        ws.cell(row=row_idx, column=1, value=away_tm).alignment = center
        ws.cell(row=row_idx, column=2, value=away_qb) 
        ws.cell(row=row_idx, column=3, value=home_tm).alignment = center
        ws.cell(row=row_idx, column=4, value=home_qb)
        
        # Vegas Input
        v_cell = ws.cell(row=row_idx, column=6, value=vegas_val)
        v_cell.fill = input_fill
        v_cell.alignment = center
        if vegas_val != "N/A": v_cell.number_format = '0.0'

        qb_dv.add(ws.cell(row=row_idx, column=2))
        qb_dv.add(ws.cell(row=row_idx, column=4))

        # Lookups
        ws.cell(row=row_idx, column=26, value=f"=VLOOKUP(A{row_idx},'DB_Teams'!A:B,2,FALSE)") 
        ws.cell(row=row_idx, column=27, value=f"=VLOOKUP(B{row_idx},'DB_QBs'!A:B,2,FALSE)")   
        ws.cell(row=row_idx, column=28, value=f"=VLOOKUP(C{row_idx},'DB_Teams'!A:B,2,FALSE)") 
        ws.cell(row=row_idx, column=29, value=f"=VLOOKUP(D{row_idx},'DB_QBs'!A:B,2,FALSE)")   
        
        # MODEL FORMULA: (Home + HFA) - Away
        # Neg - Pos = BIG NEGATIVE
        model_formula = f"=(AB{row_idx}+AC{row_idx}+Settings!$B$2) - (Z{row_idx}+AA{row_idx})"
        m_cell = ws.cell(row=row_idx, column=5, value=model_formula)
        m_cell.number_format = '0.0'
        m_cell.alignment = center
        m_cell.font = Font(bold=True)

        # Edge
        if vegas_val != "N/A":
            ws.cell(row=row_idx, column=7, value=f"=E{row_idx}-F{row_idx}").number_format = '0.0'
        else:
            ws.cell(row=row_idx, column=7, value="--")
        
        # Signal
        threshold_ref = "Settings!$B$3"
        rec_formula = (
            f'=IF(ISBLANK(F{row_idx}), "Enter Line", '
            f'IF(G{row_idx} < -{threshold_ref}, "BET " & C{row_idx}, ' 
            f'IF(G{row_idx} > {threshold_ref}, "BET " & A{row_idx}, ' 
            f'"PASS")))'
        )
        ws.cell(row=row_idx, column=8, value=rec_formula).font = Font(bold=True)
        
        row_idx += 1

    # Formatting
    ws.column_dimensions['B'].width = 25
    ws.column_dimensions['D'].width = 25
    ws.column_dimensions['H'].width = 20
    
    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    green_font = Font(color="006100", bold=True)
    ws.conditional_formatting.add(f"H2:H{row_idx}", CellIsRule(operator="beginsWith", formula=['"BET"'], stopIfTrue=True, fill=green_fill, font=green_font))

    wb['DB_Teams'].sheet_state = 'hidden'
    wb['DB_QBs'].sheet_state = 'hidden'
    wb['Settings'].sheet_state = 'hidden'

    # 3. Save Final File to Buffer
    final_buffer = io.BytesIO()
    wb.save(final_buffer)
    final_buffer.seek(0)

    st.download_button(
        label="📥 Download Interactive Betting Tool",
        data=final_buffer,
        file_name=f"NFL_Betting_Tool_Week_{current_week}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Preview Ratings
    st.subheader("Current QB Ratings")
    st.dataframe(df_qbs[['QB', 'Rating']].head(20), height=300)

except Exception as e:
    st.error(f"Error: {e}")
