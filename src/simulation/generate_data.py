"""
KDAC-4: ICC T20 World Cup 2026 — Data Simulation Module
Generates synthetic but statistically realistic cricket datasets.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─── Constants ────────────────────────────────────────────────────────────────
T20_TEAMS = [
    "India", "Australia", "England", "Pakistan", "South Africa",
    "New Zealand", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan",
    "Zimbabwe", "Ireland", "Scotland", "Netherlands", "USA",
    "Namibia", "Nepal", "UAE", "Canada", "Papua New Guinea"
]

VENUES = [
    {"name": "Eden Gardens, Kolkata",       "country": "India",       "avg_score": 168, "pitch": "spin-friendly"},
    {"name": "MCG, Melbourne",              "country": "Australia",   "avg_score": 162, "pitch": "pace-friendly"},
    {"name": "Lord's, London",              "country": "England",     "avg_score": 155, "pitch": "swing-friendly"},
    {"name": "Gaddafi Stadium, Lahore",     "country": "Pakistan",    "avg_score": 171, "pitch": "batting-friendly"},
    {"name": "Newlands, Cape Town",         "country": "S.Africa",    "avg_score": 158, "pitch": "pace-friendly"},
    {"name": "Wankhede, Mumbai",            "country": "India",       "avg_score": 174, "pitch": "batting-friendly"},
    {"name": "Nassau County, New York",     "country": "USA",         "avg_score": 149, "pitch": "pace-friendly"},
    {"name": "Sabina Park, Kingston",       "country": "West Indies", "avg_score": 161, "pitch": "pace-friendly"},
    {"name": "Dubai International Stadium", "country": "UAE",         "avg_score": 156, "pitch": "spin-friendly"},
    {"name": "Kennington Oval, London",     "country": "England",     "avg_score": 163, "pitch": "swing-friendly"},
]

BATTING_POSITIONS = list(range(1, 12))

def generate_teams_table() -> pd.DataFrame:
    """Generates team profiles with historical T20I stats."""
    teams = []
    for i, team in enumerate(T20_TEAMS):
        ranking = i + 1
        win_rate = max(0.25, 0.82 - ranking * 0.025 + np.random.normal(0, 0.03))
        teams.append({
            "team_id": i + 1,
            "team_name": team,
            "icc_ranking": ranking,
            "win_rate_last_2_years": round(win_rate, 3),
            "avg_team_score_t20": int(np.random.normal(155 - ranking, 8)),
            "avg_wickets_per_match": round(np.random.uniform(4.5, 8.5), 1),
            "nrr": round(np.random.normal(0.3 - ranking * 0.05, 0.2), 3),
            "home_win_advantage": round(np.random.uniform(0.05, 0.18), 3),
        })
    return pd.DataFrame(teams)

def generate_players_table(teams_df: pd.DataFrame) -> pd.DataFrame:
    """Generates player profiles for all teams."""
    players = []
    pid = 1
    roles = ["Batsman", "Bowler", "All-rounder", "Wicket-keeper"]
    role_weights = [0.4, 0.35, 0.15, 0.1]

    for _, team in teams_df.iterrows():
        for pos in range(1, 12):
            role = random.choices(roles, weights=role_weights)[0]
            batting_avg = round(np.random.normal(28 - pos * 1.5, 8), 1) if role != "Bowler" else round(np.random.normal(12, 5), 1)
            strike_rate = round(np.random.normal(135 - pos * 2, 18), 1)
            bowling_avg = round(np.random.normal(25, 7), 1) if role in ["Bowler", "All-rounder"] else None
            economy = round(np.random.normal(7.8, 1.2), 2) if role in ["Bowler", "All-rounder"] else None
            players.append({
                "player_id": pid,
                "team_id": team["team_id"],
                "team_name": team["team_name"],
                "player_name": f"Player_{pid}",
                "role": role,
                "batting_position": pos,
                "batting_avg": max(5.0, batting_avg),
                "strike_rate": max(80.0, strike_rate),
                "bowling_avg": bowling_avg,
                "economy_rate": economy,
                "recent_form_score": round(np.random.uniform(30, 90), 1),  # composite form metric
                "matches_played": int(np.random.randint(15, 120)),
                "is_captain": pos == 1 and random.random() < 0.3,
            })
            pid += 1
    return pd.DataFrame(players)

def generate_matches_table(teams_df: pd.DataFrame, n_matches: int = 500) -> pd.DataFrame:
    """Generates historical T20 match results."""
    matches = []
    team_ids = teams_df["team_id"].tolist()
    start_date = datetime(2021, 1, 1)

    for mid in range(1, n_matches + 1):
        team_a_id, team_b_id = random.sample(team_ids, 2)
        venue = random.choice(VENUES)
        match_date = start_date + timedelta(days=random.randint(0, 1700))

        ta = teams_df[teams_df["team_id"] == team_a_id].iloc[0]
        tb = teams_df[teams_df["team_id"] == team_b_id].iloc[0]

        # Win probability influenced by ranking + home advantage
        rank_diff = tb["icc_ranking"] - ta["icc_ranking"]
        home_bonus = ta["home_win_advantage"] if venue["country"] == ta["team_name"] else 0
        base_prob = 0.5 + rank_diff * 0.015 + home_bonus
        win_prob_a = max(0.1, min(0.9, base_prob + np.random.normal(0, 0.08)))

        winner_id = team_a_id if random.random() < win_prob_a else team_b_id
        toss_winner = random.choice([team_a_id, team_b_id])
        toss_decision = random.choices(["bat", "field"], weights=[0.45, 0.55])[0]

        score_a = int(np.random.normal(venue["avg_score"], 18))
        score_b = int(np.random.normal(venue["avg_score"], 18))

        if winner_id == team_a_id:
            score_a = max(score_b + 1, score_a)
        else:
            score_b = max(score_a + 1, score_b)

        matches.append({
            "match_id": mid,
            "date": match_date.strftime("%Y-%m-%d"),
            "venue_name": venue["name"],
            "venue_country": venue["country"],
            "pitch_type": venue["pitch"],
            "team_a_id": team_a_id,
            "team_b_id": team_b_id,
            "team_a_name": ta["team_name"],
            "team_b_name": tb["team_name"],
            "toss_winner_id": toss_winner,
            "toss_decision": toss_decision,
            "score_team_a": score_a,
            "score_team_b": score_b,
            "winner_id": winner_id,
            "winner_name": ta["team_name"] if winner_id == team_a_id else tb["team_name"],
            "win_by_runs": max(0, score_a - score_b) if winner_id == team_a_id else 0,
            "win_by_wickets": random.randint(1, 7) if winner_id == team_b_id else 0,
            "tournament": random.choice(["T20 WC 2021", "T20 WC 2022", "T20 WC 2024", "ICC Super Series 2023"]),
            "phase": random.choice(["Group Stage", "Super 8", "Semi-Final", "Final"]),
        })
    return pd.DataFrame(matches)

def generate_ball_by_ball(matches_df: pd.DataFrame, sample_matches: int = 50) -> pd.DataFrame:
    """Generates ball-by-ball event data for a sample of matches."""
    events = []
    sample = matches_df.sample(min(sample_matches, len(matches_df)), random_state=42)
    eid = 1

    for _, match in sample.iterrows():
        for inning in [1, 2]:
            team_batting = match["team_a_name"] if inning == 1 else match["team_b_name"]
            over_score = 0
            wickets = 0
            for over in range(20):
                for ball in range(6):
                    if wickets >= 10:
                        break
                    run_probs = [0.25, 0.28, 0.18, 0.10, 0.06, 0.03, 0.07, 0.03]
                    runs = random.choices([0,1,2,3,4,5,6,"W"], weights=run_probs)[0]
                    is_wicket = runs == "W"
                    actual_runs = 0 if is_wicket else int(runs)
                    if is_wicket:
                        wickets += 1
                    over_score += actual_runs
                    events.append({
                        "event_id": eid,
                        "match_id": match["match_id"],
                        "inning": inning,
                        "over": over + 1,
                        "ball": ball + 1,
                        "batting_team": team_batting,
                        "runs_off_bat": actual_runs,
                        "is_wicket": int(is_wicket),
                        "wicket_type": random.choice(["caught", "bowled", "lbw", "run out", "stumped"]) if is_wicket else None,
                        "cumulative_runs": over_score,
                        "wickets_fallen": wickets,
                    })
                    eid += 1

    return pd.DataFrame(events)

def generate_venue_conditions(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Generates venue + weather condition records linked to matches."""
    conditions = []
    for _, match in matches_df.iterrows():
        conditions.append({
            "match_id": match["match_id"],
            "venue_name": match["venue_name"],
            "temperature_c": round(np.random.normal(27, 6), 1),
            "humidity_pct": round(np.random.uniform(40, 90), 1),
            "wind_speed_kmh": round(np.random.uniform(5, 35), 1),
            "dew_factor": random.choice(["None", "Low", "Medium", "High"]),
            "pitch_pace_rating": round(np.random.uniform(3, 9), 1),
            "pitch_spin_rating": round(np.random.uniform(3, 9), 1),
            "outfield_condition": random.choice(["Excellent", "Good", "Average"]),
        })
    return pd.DataFrame(conditions)

def simulate_all(output_dir: str = "data/raw"):
    """Run full simulation and save all datasets."""
    os.makedirs(output_dir, exist_ok=True)
    print(" Simulating cricket datasets...")

    teams = generate_teams_table()
    teams.to_csv(f"{output_dir}/teams.csv", index=False)
    print(f"   teams.csv — {len(teams)} rows")

    players = generate_players_table(teams)
    players.to_csv(f"{output_dir}/players.csv", index=False)
    print(f"   players.csv — {len(players)} rows")

    matches = generate_matches_table(teams, n_matches=500)
    matches.to_csv(f"{output_dir}/matches.csv", index=False)
    print(f"   matches.csv — {len(matches)} rows")

    ball_by_ball = generate_ball_by_ball(matches, sample_matches=50)
    ball_by_ball.to_csv(f"{output_dir}/ball_by_ball.csv", index=False)
    print(f"   ball_by_ball.csv — {len(ball_by_ball)} rows")

    conditions = generate_venue_conditions(matches)
    conditions.to_csv(f"{output_dir}/venue_conditions.csv", index=False)
    print(f"   venue_conditions.csv — {len(conditions)} rows")

    print("\n Simulation complete. Files saved to:", output_dir)
    return teams, players, matches, ball_by_ball, conditions

if __name__ == "__main__":
    simulate_all()
