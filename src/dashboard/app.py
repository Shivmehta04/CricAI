"""
KDAC-4: Streamlit Dashboard
EDA visualizations + Live match prediction interface
Run: streamlit run src/dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

from src.models.predict import predict_match, get_team_stats
from src.genai.strategy import generate_strategy

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CricAI — T20 WC 2026 Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = "data/cricket_warehouse.db"
MODEL_PATH = "data/models/best_model.pkl"

def ensure_initialized():
    """Ensure data simulation, ETL, and ML training are run if missing."""
    if not os.path.exists(DB_PATH) or not os.path.exists(MODEL_PATH):
        st.warning("⚡ First-time initialization running data pipeline & ML training...")
        from src.simulation.generate_data import simulate_all
        from src.etl.pipeline import run_pipeline
        from src.models.train_model import run_training

        simulate_all(output_dir="data/raw")
        run_pipeline()
        run_training()
        st.success("✅ Initialization complete!")

ensure_initialized()

@st.cache_data
def load_table(table: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/44/ICC_logo.svg/200px-ICC_logo.svg.png", width=80)
st.sidebar.title("🏏 CricAI Platform")
st.sidebar.markdown("**KDAC-4 | ICC T20 WC 2026**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 EDA Dashboard",
    "🤖 Match Predictor",
    "📋 Strategy Brief",
    "🗄️ Data Warehouse",
])

# ─── PAGE: Overview ───────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🏏 CricAI — ICC T20 World Cup 2026 Prediction Platform")
    st.markdown("### End-to-end cricket analytics: Data → ETL → ML → Strategy")

    col1, col2, col3, col4 = st.columns(4)
    try:
        teams = load_table("teams")
        matches = load_table("matches")
        players = load_table("players")

        col1.metric("🌍 Teams", len(teams))
        col2.metric("🏟️ Matches Analyzed", len(matches))
        col3.metric("👤 Players Profiled", len(players))
        col4.metric("📅 Years Covered", "2021–2026")
    except Exception as e:
        st.error(f"Database not ready: {e}. Please run `python run_demo.py` first.")

    st.markdown("---")
    st.subheader("Platform Architecture")
    st.image("https://i.imgur.com/placeholder.png", use_container_width=True,
             caption="CricAI End-to-End Pipeline")

    st.subheader("How It Works")
    cols = st.columns(6)
    steps = [
        ("1️⃣", "Data Simulation", "Synthetic T20 datasets with real distributions"),
        ("2️⃣", "ETL Pipeline", "Clean, transform, load into SQLite warehouse"),
        ("3️⃣", "EDA Dashboards", "Visual exploration of trends & patterns"),
        ("4️⃣", "ML Models", "XGBoost + RF ensemble for win prediction"),
        ("5️⃣", "Prediction", "Real-time win probability with confidence score"),
        ("6️⃣", "GenAI Strategy", "LLM-powered plain-language coaching notes"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(f"**{num} {title}**")
        col.caption(desc)

# ─── PAGE: EDA Dashboard ──────────────────────────────────────────────────────
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")

    try:
        matches = load_table("matches")
        teams = load_table("teams")
        players = load_table("players")
        matches["date"] = pd.to_datetime(matches["date"])

        tab1, tab2, tab3, tab4 = st.tabs(["Team Performance", "Venue Analysis", "Toss Analysis", "Player Stats"])

        with tab1:
            st.subheader("Win Rate by Team")
            fig = px.bar(
                teams.sort_values("win_rate_last_2_years", ascending=False),
                x="team_name", y="win_rate_last_2_years",
                color="win_rate_last_2_years",
                color_continuous_scale="Greens",
                labels={"win_rate_last_2_years": "Win Rate", "team_name": "Team"},
                title="Team Win Rates (Last 2 Years)"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Average Team Score vs ICC Ranking")
            fig2 = px.scatter(
                teams, x="icc_ranking", y="avg_team_score_t20",
                size="win_rate_last_2_years", color="team_name",
                hover_name="team_name",
                title="ICC Ranking vs Average T20 Score (bubble = win rate)"
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.subheader("Average Score by Venue")
            venue_scores = matches.groupby("venue_name")["score_team_a"].mean().reset_index()
            venue_scores.columns = ["venue", "avg_score"]
            venue_scores = venue_scores.sort_values("avg_score", ascending=True)
            fig3 = px.bar(venue_scores, x="avg_score", y="venue",
                          orientation="h", color="avg_score",
                          color_continuous_scale="Blues",
                          title="Average First-Innings Score by Venue")
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Matches by Pitch Type")
            pitch_counts = matches["pitch_type"].value_counts().reset_index()
            fig4 = px.pie(pitch_counts, values="count", names="pitch_type",
                          title="Distribution of Pitch Types")
            st.plotly_chart(fig4, use_container_width=True)

        with tab3:
            st.subheader("Toss Decision Frequency")
            toss_counts = matches["toss_decision"].value_counts().reset_index()
            fig5 = px.pie(toss_counts, values="count", names="toss_decision",
                          title="Toss Decision: Bat vs Field")
            st.plotly_chart(fig5, use_container_width=True)

            st.subheader("Does Winning the Toss Help?")
            matches["toss_win_match"] = (
                matches["toss_winner_id"] == matches["winner_id"]
            ).astype(int)
            toss_effect = matches["toss_win_match"].value_counts().reset_index()
            toss_effect["label"] = toss_effect["toss_win_match"].map({1: "Toss Winner Won", 0: "Toss Winner Lost"})
            fig6 = px.bar(toss_effect, x="label", y="count", color="label",
                          title="Toss Winner Match Win Rate")
            st.plotly_chart(fig6, use_container_width=True)

        with tab4:
            st.subheader("Player Role Distribution")
            role_counts = players["role"].value_counts().reset_index()
            fig7 = px.pie(role_counts, values="count", names="role",
                          title="Player Role Distribution Across All Teams")
            st.plotly_chart(fig7, use_container_width=True)

            st.subheader("Top 15 Players by Batting Average")
            top_bat = players[players["role"] != "Bowler"].nlargest(15, "batting_avg")
            fig8 = px.bar(top_bat, x="player_name", y="batting_avg",
                          color="team_name", title="Top 15 Batsmen by Average")
            st.plotly_chart(fig8, use_container_width=True)

    except Exception as e:
        st.error(f"Data not loaded: {e}. Run `python run_demo.py` first.")

# ─── PAGE: Match Predictor ────────────────────────────────────────────────────
elif page == "🤖 Match Predictor":
    st.title("🤖 Match Outcome Predictor")
    st.markdown("Configure the match conditions below and get an AI-powered prediction.")

    try:
        teams = load_table("teams")
        team_names = sorted(teams["team_name"].tolist())

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Team Selection")
            team_a = st.selectbox("🔵 Team A", team_names, index=0)
            team_b = st.selectbox("🔴 Team B", [t for t in team_names if t != team_a], index=1)
            toss_winner = st.selectbox("🪙 Toss Winner", [team_a, team_b])
            toss_decision = st.radio("Toss Decision", ["bat", "field"], horizontal=True)

        with col2:
            st.subheader("Match Conditions")
            phase = st.selectbox("Tournament Phase", ["Group Stage", "Super 8", "Semi-Final", "Final"])
            pitch_type = st.selectbox("Pitch Type", ["batting-friendly", "spin-friendly", "pace-friendly", "swing-friendly"])
            temperature = st.slider("Temperature (°C)", 15, 45, 28)
            humidity = st.slider("Humidity (%)", 30, 100, 65)
            dew = st.select_slider("Dew Factor", ["None", "Low", "Medium", "High"], value="Low")
            dew_num = {"None": 0, "Low": 1, "Medium": 2, "High": 3}[dew]

        if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
            with st.spinner("Running prediction model..."):
                result = predict_match(
                    team_a, team_b,
                    toss_winner=toss_winner,
                    toss_decision=toss_decision,
                    pitch_type=pitch_type,
                    phase=phase,
                    temperature_c=temperature,
                    humidity_pct=humidity,
                    dew_factor_numeric=dew_num,
                )

            st.session_state["last_result"] = result
            st.session_state["last_dew"] = dew

            st.success(f"✅ Prediction Complete!")
            st.markdown("---")

            # Results display
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("🏆 Predicted Winner", result["predicted_winner"])
            rc2.metric(f"{team_a} Win Prob", f"{result['win_prob_a']}%")
            rc3.metric(f"{team_b} Win Prob", f"{result['win_prob_b']}%")

            # Probability bar
            fig = go.Figure(go.Bar(
                x=[result["win_prob_a"], result["win_prob_b"]],
                y=[team_a, team_b],
                orientation="h",
                marker_color=["#1a7a4a", "#d73027"],
                text=[f"{result['win_prob_a']}%", f"{result['win_prob_b']}%"],
                textposition="auto"
            ))
            fig.update_layout(title="Win Probability Distribution", height=200, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔑 Key Prediction Factors")
            for f in result["top_factors"]:
                st.markdown(f"- {f}")

            confidence = result["confidence_score"]
            color = "green" if confidence > 60 else "orange" if confidence > 35 else "red"
            st.markdown(f"**Model Confidence:** :{color}[{confidence}/100]")

    except Exception as e:
        st.error(f"Model not ready: {e}. Run `python run_demo.py` first.")

# ─── PAGE: Strategy Brief ─────────────────────────────────────────────────────
elif page == "📋 Strategy Brief":
    st.title("📋 GenAI Strategy Brief")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        dew = st.session_state.get("last_dew", "Low")
        st.info(f"Generating strategy for: **{result['team_a']} vs {result['team_b']}**")
        strategy = generate_strategy(result, dew_level=dew)
        st.code(strategy, language=None)
        st.download_button("📥 Download Strategy Brief", strategy,
                           file_name=f"strategy_{result['team_a']}_vs_{result['team_b']}.txt")
    else:
        st.warning("Run a prediction first from the **Match Predictor** page.")

# ─── PAGE: Data Warehouse ─────────────────────────────────────────────────────
elif page == "🗄️ Data Warehouse":
    st.title("🗄️ Data Warehouse Explorer")
    table_choice = st.selectbox("Select Table", ["teams", "players", "matches", "ml_features", "innings_summary"])
    try:
        df = load_table(table_choice)
        st.write(f"**{len(df)} rows × {len(df.columns)} columns**")
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button(f"📥 Download {table_choice}.csv",
                           df.to_csv(index=False),
                           file_name=f"{table_choice}.csv")
    except Exception as e:
        st.error(f"Table not found: {e}")
