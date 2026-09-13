"""
KDAC-4: Match Outcome Prediction Engine
Loads trained model and predicts win probability for any Team A vs Team B matchup.
"""

import pickle
import numpy as np
import pandas as pd
import os
import sqlite3

MODEL_DIR = "data/models"
DB_PATH = "data/cricket_warehouse.db"

PITCH_TYPES = ["batting-friendly", "spin-friendly", "pace-friendly", "swing-friendly"]
PHASES = ["Group Stage", "Super 8", "Semi-Final", "Final"]
TOSS_DECISIONS = ["bat", "field"]

_model_cache = {}

def _load_artifacts():
    """Load model and metadata (cached)."""
    if _model_cache:
        return _model_cache

    model_path = f"{MODEL_DIR}/best_model.pkl"
    meta_path = f"{MODEL_DIR}/metadata.pkl"
    le_path = f"{MODEL_DIR}/label_encoders.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not trained yet. Run train_model.py first.")

    with open(model_path, "rb") as f:
        _model_cache["model"] = pickle.load(f)
    with open(meta_path, "rb") as f:
        _model_cache["meta"] = pickle.load(f)
    with open(le_path, "rb") as f:
        _model_cache["le"] = pickle.load(f)
    return _model_cache


def get_team_stats(team_name: str, db_path: str = DB_PATH) -> dict:
    """Retrieve team stats from the warehouse."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM teams WHERE team_name = ? LIMIT 1",
        conn, params=(team_name,)
    )
    conn.close()
    if df.empty:
        raise ValueError(f"Team '{team_name}' not found in warehouse.")
    return df.iloc[0].to_dict()


def predict_match(
    team_a: str,
    team_b: str,
    toss_winner: str = None,
    toss_decision: str = "field",
    pitch_type: str = "batting-friendly",
    phase: str = "Group Stage",
    temperature_c: float = 28.0,
    humidity_pct: float = 65.0,
    wind_speed_kmh: float = 15.0,
    dew_factor_numeric: int = 1,
    pitch_pace_rating: float = 6.0,
    pitch_spin_rating: float = 5.0,
) -> dict:
    """
    Predict match outcome probability for Team A vs Team B.

    Returns:
        dict with win_prob_a, win_prob_b, confidence, top_factors
    """
    arts = _load_artifacts()
    model = arts["model"]
    meta = arts["meta"]
    le_dict = arts["le"]
    feature_cols = meta["feature_cols"]

    ta = get_team_stats(team_a)
    tb = get_team_stats(team_b)

    # Build raw feature row
    toss_win = 1 if (toss_winner == team_a or toss_winner is None) else 0

    row = {
        "rank_diff": ta["icc_ranking"] - tb["icc_ranking"],
        "win_rate_diff": ta["win_rate_last_2_years"] - tb["win_rate_last_2_years"],
        "avg_score_diff": ta["avg_team_score_t20"] - tb["avg_team_score_t20"],
        "nrr_diff": ta["nrr"] - tb["nrr"],
        "ta_icc_ranking": ta["icc_ranking"],
        "tb_icc_ranking": tb["icc_ranking"],
        "ta_win_rate_last_2_years": ta["win_rate_last_2_years"],
        "tb_win_rate_last_2_years": tb["win_rate_last_2_years"],
        "ta_avg_team_score_t20": ta["avg_team_score_t20"],
        "tb_avg_team_score_t20": tb["avg_team_score_t20"],
        "toss_win": toss_win,
        "toss_decision": toss_decision,
        "pitch_type": pitch_type,
        "phase": phase,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "wind_speed_kmh": wind_speed_kmh,
        "dew_factor_numeric": dew_factor_numeric,
        "pitch_pace_rating": pitch_pace_rating,
        "pitch_spin_rating": pitch_spin_rating,
    }

    # Encode categoricals
    for col, le in le_dict.items():
        if col in row:
            try:
                row[col] = le.transform([str(row[col])])[0]
            except ValueError:
                row[col] = 0  # unseen label fallback

    # Build feature vector
    X = pd.DataFrame([row])[feature_cols].fillna(0)

    prob = model.predict_proba(X)[0]
    win_prob_a = float(prob[1])
    win_prob_b = float(prob[0])

    # Confidence = distance from 0.5
    confidence = abs(win_prob_a - 0.5) * 200  # 0-100 scale

    # Top factors (human-readable)
    top_factors = _build_top_factors(ta, tb, row, win_prob_a)

    predicted_winner = team_a if win_prob_a >= 0.5 else team_b

    return {
        "team_a": team_a,
        "team_b": team_b,
        "predicted_winner": predicted_winner,
        "win_prob_a": round(win_prob_a * 100, 1),
        "win_prob_b": round(win_prob_b * 100, 1),
        "confidence_score": round(confidence, 1),
        "top_factors": top_factors,
        "pitch_type": pitch_type,
        "phase": phase,
    }


def _build_top_factors(ta, tb, row, win_prob_a):
    """Generate human-readable key prediction drivers."""
    factors = []

    rank_diff = ta["icc_ranking"] - tb["icc_ranking"]
    if abs(rank_diff) >= 2:
        better = ta if rank_diff < 0 else tb
        factors.append(f"ICC Ranking advantage: {better['team_name']} ranked #{better['icc_ranking']}")

    wr_diff = ta["win_rate_last_2_years"] - tb["win_rate_last_2_years"]
    if abs(wr_diff) > 0.05:
        better = ta if wr_diff > 0 else tb
        factors.append(f"Recent form: {better['team_name']} win rate {better['win_rate_last_2_years']:.0%}")

    if row.get("toss_win", 0) == 1:
        factors.append(f"Toss advantage: Toss-winner elected to {row.get('toss_decision', 'field')}")

    if row.get("dew_factor_numeric", 0) >= 2:
        factors.append("Heavy dew: favours chasing team (second innings)")

    pitch = row.get("pitch_type", "")
    if isinstance(pitch, int):
        pitch = "unknown"
    factors.append(f"Pitch condition: {pitch}")

    if win_prob_a > 0.65:
        factors.append(f"{ta['team_name']} hold strong head-to-head advantage at this phase")
    elif win_prob_a < 0.35:
        factors.append(f"{tb['team_name']} statistically superior across all key metrics")

    return factors[:4]


if __name__ == "__main__":
    result = predict_match("India", "Australia", phase="Semi-Final")
    print("\n" + "=" * 50)
    print(f"  {result['team_a']} vs {result['team_b']}")
    print("=" * 50)
    print(f"  Predicted Winner : {result['predicted_winner']}")
    print(f"  Win Probability  : {result['team_a']} {result['win_prob_a']}%  |  {result['team_b']} {result['win_prob_b']}%")
    print(f"  Confidence Score : {result['confidence_score']}/100")
    print("\n  Key Factors:")
    for f in result["top_factors"]:
        print(f"    • {f}")
