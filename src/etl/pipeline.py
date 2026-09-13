"""
KDAC-4: ETL Pipeline
Extract  Transform  Load cricket data into a clean, analysis-ready warehouse.
"""

import pandas as pd
import numpy as np
import os
import sqlite3

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
DB_PATH = "data/cricket_warehouse.db"

# ─── EXTRACT ──────────────────────────────────────────────────────────────────

def extract(raw_dir: str = RAW_DIR) -> dict:
    """Load all raw CSVs into dataframes."""
    print("\n [ETL] EXTRACT phase...")
    tables = {}
    for fname in ["teams", "players", "matches", "ball_by_ball", "venue_conditions"]:
        path = os.path.join(raw_dir, f"{fname}.csv")
        if os.path.exists(path):
            tables[fname] = pd.read_csv(path)
            print(f"  Loaded {fname}: {len(tables[fname])} rows")
        else:
            print(f"   Missing: {path}")
    return tables

# ─── TRANSFORM ────────────────────────────────────────────────────────────────

def transform(tables: dict) -> dict:
    """Clean, validate, and engineer features from raw data."""
    print("\n [ETL] TRANSFORM phase...")
    processed = {}

    # ── Teams ──────────────────────────────────────────────────────────────────
    teams = tables["teams"].copy()
    teams["win_rate_last_2_years"] = teams["win_rate_last_2_years"].clip(0, 1)
    teams["avg_team_score_t20"] = teams["avg_team_score_t20"].clip(100, 220)
    processed["teams"] = teams
    print(f"   teams: {len(teams)} records cleaned")

    # ── Players ────────────────────────────────────────────────────────────────
    players = tables["players"].copy()
    players["batting_avg"] = players["batting_avg"].clip(0, 100)
    players["strike_rate"] = players["strike_rate"].clip(50, 250)
    players["bowling_avg"] = players["bowling_avg"].fillna(999)
    players["economy_rate"] = players["economy_rate"].fillna(999)
    players["recent_form_score"] = players["recent_form_score"].clip(0, 100)
    # Composite player value index
    players["player_value_index"] = (
        players["batting_avg"] * 0.4 +
        players["strike_rate"] * 0.1 +
        players["recent_form_score"] * 0.5
    ).round(2)
    processed["players"] = players
    print(f"   players: {len(players)} records cleaned + player_value_index computed")

    # ── Matches ────────────────────────────────────────────────────────────────
    matches = tables["matches"].copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches["match_year"] = matches["date"].dt.year
    matches["match_month"] = matches["date"].dt.month
    # Derived columns
    matches["run_diff"] = matches["score_team_a"] - matches["score_team_b"]
    matches["total_runs"] = matches["score_team_a"] + matches["score_team_b"]
    matches["team_a_won"] = (matches["winner_id"] == matches["team_a_id"]).astype(int)
    # Toss effect
    matches["toss_winner_batted"] = (
        (matches["toss_winner_id"] == matches["team_a_id"]) &
        (matches["toss_decision"] == "bat")
    ).astype(int)
    processed["matches"] = matches
    print(f"   matches: {len(matches)} records cleaned + features derived")

    # ── Venue conditions ────────────────────────────────────────────────────────
    conditions = tables["venue_conditions"].copy()
    dew_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    conditions["dew_factor_numeric"] = conditions["dew_factor"].map(dew_map)
    processed["venue_conditions"] = conditions
    print(f"   venue_conditions: {len(conditions)} records cleaned")

    # ── Ball-by-ball aggregation ────────────────────────────────────────────────
    bbb = tables["ball_by_ball"].copy()
    bbb_agg = bbb.groupby(["match_id", "inning"]).agg(
        total_runs=("runs_off_bat", "sum"),
        total_wickets=("is_wicket", "sum"),
        total_balls=("ball", "count"),
        boundaries=("runs_off_bat", lambda x: (x >= 4).sum()),
        sixes=("runs_off_bat", lambda x: (x == 6).sum()),
    ).reset_index()
    bbb_agg["run_rate"] = (bbb_agg["total_runs"] / bbb_agg["total_balls"] * 6).round(2)
    processed["innings_summary"] = bbb_agg
    print(f"   innings_summary: {len(bbb_agg)} records aggregated from ball-by-ball")

    # ── Master feature table for ML ────────────────────────────────────────────
    processed["ml_features"] = _build_ml_features(matches, teams, conditions)

    return processed

def _build_ml_features(matches: pd.DataFrame, teams: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix for ML model training."""
    # Merge team stats for team A
    feat = matches.merge(
        teams.add_prefix("ta_").rename(columns={"ta_team_id": "team_a_id"}),
        on="team_a_id", how="left"
    )
    # Merge team stats for team B
    feat = feat.merge(
        teams.add_prefix("tb_").rename(columns={"tb_team_id": "team_b_id"}),
        on="team_b_id", how="left"
    )
    # Merge venue conditions
    feat = feat.merge(conditions, on="match_id", how="left")

    # Feature engineering
    feat["rank_diff"] = feat["ta_icc_ranking"] - feat["tb_icc_ranking"]
    feat["win_rate_diff"] = feat["ta_win_rate_last_2_years"] - feat["tb_win_rate_last_2_years"]
    feat["avg_score_diff"] = feat["ta_avg_team_score_t20"] - feat["tb_avg_team_score_t20"]
    feat["nrr_diff"] = feat["ta_nrr"] - feat["tb_nrr"]
    feat["toss_win"] = (feat["toss_winner_id"] == feat["team_a_id"]).astype(int)

    feature_cols = [
        "match_id", "team_a_name", "team_b_name",
        "rank_diff", "win_rate_diff", "avg_score_diff", "nrr_diff",
        "ta_icc_ranking", "tb_icc_ranking",
        "ta_win_rate_last_2_years", "tb_win_rate_last_2_years",
        "ta_avg_team_score_t20", "tb_avg_team_score_t20",
        "toss_win", "toss_decision",
        "temperature_c", "humidity_pct", "wind_speed_kmh",
        "dew_factor_numeric", "pitch_pace_rating", "pitch_spin_rating",
        "pitch_type", "match_year", "phase",
        "team_a_won"  # target
    ]
    existing_cols = [c for c in feature_cols if c in feat.columns]
    ml_feat = feat[existing_cols].dropna(subset=["team_a_won"])
    print(f"   ml_features: {len(ml_feat)} rows, {len(ml_feat.columns)} columns")
    return ml_feat

# ─── LOAD ─────────────────────────────────────────────────────────────────────

def load(processed: dict, db_path: str = DB_PATH, csv_dir: str = PROCESSED_DIR):
    """Load processed data into SQLite warehouse + CSV files."""
    print(f"\n [ETL] LOAD phase  {db_path}")
    os.makedirs(csv_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)

    for table_name, df in processed.items():
        # Save to SQLite
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        # Save to CSV
        df.to_csv(os.path.join(csv_dir, f"{table_name}.csv"), index=False)
        print(f"   Loaded: {table_name} ({len(df)} rows)")

    conn.close()
    print(f"\n ETL complete. Warehouse: {db_path}")

# ─── PIPELINE ─────────────────────────────────────────────────────────────────

def run_pipeline():
    """Run full ETL pipeline: Extract  Transform  Load."""
    print("=" * 60)
    print("  KDAC-4 ETL PIPELINE — Cricket Data Warehouse Builder")
    print("=" * 60)

    tables = extract()
    if not tables:
        print(" No raw data found. Run simulate first.")
        return None

    processed = transform(tables)
    load(processed)
    return processed

if __name__ == "__main__":
    run_pipeline()
