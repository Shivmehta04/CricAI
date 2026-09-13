"""
KDAC-4: One-Click Demo Runner
Runs the full pipeline: Simulate -> ETL -> Train -> Predict -> Strategy
Usage: python run_demo.py
"""

import os
import sys

# Force UTF-8 output to avoid cp1252 issues on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

def banner(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def main():
    banner("KDAC-4: CricAI -- ICC T20 WC 2026 Prediction Platform")
    print("  Full pipeline demo: Simulate -> ETL -> Train -> Predict -> Strategy")

    # -- Step 1: Simulate data --
    banner("STEP 1/5: Data Simulation")
    from src.simulation.generate_data import simulate_all
    simulate_all(output_dir="data/raw")

    # -- Step 2: ETL pipeline --
    banner("STEP 2/5: ETL Pipeline")
    from src.etl.pipeline import run_pipeline
    run_pipeline()

    # -- Step 3: Train model --
    banner("STEP 3/5: ML Model Training")
    from src.models.train_model import run_training
    run_training()

    # -- Step 4: Prediction demo --
    banner("STEP 4/5: Match Prediction Demo")
    from src.models.predict import predict_match

    fixtures = [
        ("India", "Pakistan", "Semi-Final", "spin-friendly"),
        ("Australia", "England", "Final", "pace-friendly"),
        ("South Africa", "New Zealand", "Super 8", "batting-friendly"),
    ]

    results = []
    for team_a, team_b, phase, pitch in fixtures:
        r = predict_match(team_a, team_b, phase=phase, pitch_type=pitch)
        results.append(r)
        print(f"\n  {team_a} vs {team_b} ({phase})")
        print(f"  -> Winner: {r['predicted_winner']}  "
              f"[{r['win_prob_a']}% vs {r['win_prob_b']}%]  "
              f"Confidence: {r['confidence_score']}/100")

    # -- Step 5: Strategy generation --
    banner("STEP 5/5: GenAI Strategy Brief (India vs Pakistan)")
    from src.genai.strategy import generate_strategy
    strategy = generate_strategy(results[0], dew_level="Medium")
    print(strategy)

    # -- Done --
    banner("DEMO COMPLETE")
    print("""
  Next steps:
  ─────────────────────────────────────────────────
  Dashboard : streamlit run src/dashboard/app.py
  API       : uvicorn src.api.main:app --reload
  API Docs  : http://localhost:8000/docs
  Docker    : docker-compose -f docker/docker-compose.yml up
  ─────────────────────────────────────────────────
    """)

if __name__ == "__main__":
    main()
