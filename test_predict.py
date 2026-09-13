"""Quick prediction test - run after run_demo.py"""
import sys
sys.path.insert(0, '.')

from src.models.predict import predict_match
from src.genai.strategy import generate_strategy

print("=" * 50)
print("  PREDICTION TEST")
print("=" * 50)

fixtures = [
    ("India", "Pakistan", "Semi-Final", "spin-friendly"),
    ("Australia", "England", "Final", "pace-friendly"),
    ("South Africa", "New Zealand", "Super 8", "batting-friendly"),
]

results = []
for team_a, team_b, phase, pitch in fixtures:
    r = predict_match(team_a, team_b, phase=phase, pitch_type=pitch)
    results.append(r)
    print(f"\n{team_a} vs {team_b} ({phase})")
    print(f"  Winner : {r['predicted_winner']}")
    print(f"  Prob   : {r['win_prob_a']}% vs {r['win_prob_b']}%")
    print(f"  Conf   : {r['confidence_score']}/100")
    for f in r['top_factors']:
        print(f"  - {f}")

print("\n" + "=" * 50)
print("  STRATEGY BRIEF: India vs Pakistan")
print("=" * 50)
strategy = generate_strategy(results[0], dew_level="Medium")
print(strategy)
