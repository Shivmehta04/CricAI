"""
KDAC-4: FastAPI Backend
REST API endpoints for match prediction and strategy generation.
Run: uvicorn src.api.main:app --reload
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sqlite3
import pandas as pd

from src.models.predict import predict_match
from src.genai.strategy import generate_strategy

app = FastAPI(
    title="CricAI API",
    description="KDAC-4: ICC T20 World Cup 2026 Match Outcome Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "data/cricket_warehouse.db"

# ─── Schemas ──────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    team_a: str = Field(..., example="India")
    team_b: str = Field(..., example="Australia")
    toss_winner: Optional[str] = Field(None, example="India")
    toss_decision: str = Field("field", example="field")
    pitch_type: str = Field("batting-friendly", example="spin-friendly")
    phase: str = Field("Group Stage", example="Semi-Final")
    temperature_c: float = Field(28.0, ge=10, le=50)
    humidity_pct: float = Field(65.0, ge=10, le=100)
    wind_speed_kmh: float = Field(15.0, ge=0, le=80)
    dew_factor_numeric: int = Field(1, ge=0, le=3)
    pitch_pace_rating: float = Field(6.0, ge=1, le=10)
    pitch_spin_rating: float = Field(5.0, ge=1, le=10)

class StrategyRequest(BaseModel):
    prediction: dict
    dew_level: str = Field("Low", example="Medium")

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "CricAI API", "version": "1.0.0"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

@app.get("/teams", tags=["Data"])
def list_teams():
    """List all available teams."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT team_name, icc_ranking, win_rate_last_2_years FROM teams ORDER BY icc_ranking", conn)
        conn.close()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/teams/{team_name}", tags=["Data"])
def get_team(team_name: str):
    """Get stats for a specific team."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM teams WHERE team_name = ?", conn, params=(team_name,))
        conn.close()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found.")
        return df.iloc[0].to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/matches/recent", tags=["Data"])
def recent_matches(limit: int = 10):
    """Get most recent matches."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            f"SELECT match_id, date, team_a_name, team_b_name, winner_name, venue_name FROM matches ORDER BY date DESC LIMIT {limit}",
            conn
        )
        conn.close()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", tags=["Prediction"])
def predict(req: PredictionRequest):
    """
    Predict match outcome between two teams.
    Returns win probabilities and key factors.
    """
    try:
        result = predict_match(
            team_a=req.team_a,
            team_b=req.team_b,
            toss_winner=req.toss_winner,
            toss_decision=req.toss_decision,
            pitch_type=req.pitch_type,
            phase=req.phase,
            temperature_c=req.temperature_c,
            humidity_pct=req.humidity_pct,
            wind_speed_kmh=req.wind_speed_kmh,
            dew_factor_numeric=req.dew_factor_numeric,
            pitch_pace_rating=req.pitch_pace_rating,
            pitch_spin_rating=req.pitch_spin_rating,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/strategy", tags=["Strategy"])
def get_strategy(req: StrategyRequest):
    """
    Generate plain-language strategy notes from a prediction result.
    """
    try:
        strategy = generate_strategy(req.prediction, dew_level=req.dew_level)
        return {"strategy": strategy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-and-strategy", tags=["Prediction"])
def predict_and_strategy(req: PredictionRequest):
    """
    One-call endpoint: predict + generate strategy brief.
    """
    try:
        result = predict_match(
            team_a=req.team_a, team_b=req.team_b,
            toss_winner=req.toss_winner, toss_decision=req.toss_decision,
            pitch_type=req.pitch_type, phase=req.phase,
            temperature_c=req.temperature_c, humidity_pct=req.humidity_pct,
            wind_speed_kmh=req.wind_speed_kmh,
            dew_factor_numeric=req.dew_factor_numeric,
            pitch_pace_rating=req.pitch_pace_rating,
            pitch_spin_rating=req.pitch_spin_rating,
        )
        dew_map = {0: "None", 1: "Low", 2: "Medium", 3: "High"}
        strategy = generate_strategy(result, dew_level=dew_map.get(req.dew_factor_numeric, "Low"))
        return {"prediction": result, "strategy": strategy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
