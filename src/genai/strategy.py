"""
KDAC-4: GenAI Strategy Layer
Converts ML prediction outputs into plain-language strategy recommendations.
Uses template-based generation (no API key needed for POC demo).
Optional: Plug in OpenAI/Gemini API for richer LLM output.
"""

import os
import random
from datetime import datetime

# ─── Template-based strategy engine (works without API key) ──────────────────

BATTING_STRATEGIES = {
    "batting-friendly": [
        "Set an aggressive target of 185+ in the first innings.",
        "Open with your most attacking batsmen to capitalize on the flat pitch.",
        "Back your top-order to score freely in the powerplay (overs 1-6).",
    ],
    "spin-friendly": [
        "Target 155-165 — slow wicket will reduce scoring potential.",
        "Use the spin-heavy middle overs to build a steady total.",
        "Promote an aggressive lower-order hitter for the final 4 overs.",
    ],
    "pace-friendly": [
        "Aim for 160-170; early wickets will be crucial on a seaming surface.",
        "Anchor the innings with your most technically sound batsmen.",
        "Accelerate only after the 12th over when surface typically flattens.",
    ],
    "swing-friendly": [
        "Survive the first 6 overs cautiously — new-ball swing will be significant.",
        "A 150+ score will be competitive on this surface.",
        "Push hard in overs 13-20 once the ball stops swinging.",
    ],
}

BOWLING_STRATEGIES = {
    "batting-friendly": [
        "Use your fastest bowlers in the powerplay to restrict early momentum.",
        "Deploy change-of-pace bowlers in overs 7-15 to create dot balls.",
        "Save your best death bowler for overs 17-20.",
    ],
    "spin-friendly": [
        "Open with your premier spinner from over 3 — surface will grip early.",
        "Use two spinners in tandem in the middle overs (7-16).",
        "Target top-order batsmen with flight and turn.",
    ],
    "pace-friendly": [
        "Bowl full and straight to exploit seam movement in overcast conditions.",
        "Use your new-ball pacer for consecutive spells while ball is hard.",
        "Mix shorter lengths to create uncertainty for batsmen.",
    ],
    "swing-friendly": [
        "Bowl with the wind in the powerplay — maximize swing.",
        "Use a reverse-swinging senior ball in overs 15-20.",
        "Set aggressive slip cordon for new batsmen.",
    ],
}

TOSS_TIPS = {
    "bat": {
        "low_dew": "Batting first is the right call — dew unlikely to affect the surface significantly.",
        "high_dew": " Toss winner chose to bat despite heavy dew — chasing team may benefit later.",
    },
    "field": {
        "low_dew": "Chasing gives information advantage but pitch conditions may deteriorate.",
        "high_dew": "Fielding first is strategically sound — dew will make bowling harder in 2nd innings.",
    }
}

PHASE_TIPS = {
    "Group Stage": "Prioritize net run rate — big wins here can decide advancement.",
    "Super 8": "Pressure intensifies; key players must be fresh and in form.",
    "Semi-Final": "Tournament experience and temperament under pressure are decisive.",
    "Final": "One game decides everything — back your best 11, no experiments.",
}

KEY_PLAYER_ROLES = [
    "Ensure your {role} bowls all 4 overs — they're your most economical option.",
    "Your {role} must convert in the middle overs; they're the hinge of the innings.",
    "A strong performance from your {role} in the first 6 overs sets the platform.",
    "Your {role} should target the opposition's anchor batsman with short-pitch deliveries.",
]


def generate_strategy(prediction_result: dict, dew_level: str = "Low") -> str:
    """
    Generate plain-language strategy notes from ML prediction output.

    Args:
        prediction_result: Output dict from predict.predict_match()
        dew_level: "None", "Low", "Medium", "High"

    Returns:
        Formatted strategy string
    """
    team_a = prediction_result["team_a"]
    team_b = prediction_result["team_b"]
    winner = prediction_result["predicted_winner"]
    loser = team_b if winner == team_a else team_a
    win_prob = prediction_result[f"win_prob_a"] if winner == team_a else prediction_result["win_prob_b"]
    confidence = prediction_result["confidence_score"]
    pitch = prediction_result.get("pitch_type", "batting-friendly")
    phase = prediction_result.get("phase", "Group Stage")
    top_factors = prediction_result.get("top_factors", [])

    dew_key = "high_dew" if dew_level in ["Medium", "High"] else "low_dew"

    batting_tips = random.sample(BATTING_STRATEGIES.get(pitch, BATTING_STRATEGIES["batting-friendly"]), 2)
    bowling_tips = random.sample(BOWLING_STRATEGIES.get(pitch, BOWLING_STRATEGIES["batting-friendly"]), 2)

    roles = ["opening bowler", "middle-order anchor", "death bowler", "spinner"]
    player_tip = random.choice(KEY_PLAYER_ROLES).format(role=random.choice(roles))

    lines = [
        "=" * 60,
        f"   STRATEGY BRIEF — {team_a.upper()} vs {team_b.upper()}",
        f"   Phase: {phase}  |  Pitch: {pitch}  |  Dew: {dew_level}",
        "=" * 60,
        "",
        f" PREDICTION SUMMARY",
        f"  Predicted Winner : {winner}",
        f"  Win Probability  : {win_prob}%  (Confidence: {confidence}/100)",
        "",
        " KEY PREDICTION DRIVERS",
    ]
    for factor in top_factors:
        lines.append(f"  • {factor}")

    lines += [
        "",
        f" BATTING STRATEGY ({winner} — if batting first)",
        f"  • {batting_tips[0]}",
        f"  • {batting_tips[1]}",
        "",
        f" BOWLING STRATEGY ({winner} — containment plan for {loser})",
        f"  • {bowling_tips[0]}",
        f"  • {bowling_tips[1]}",
        "",
        f"🪙 TOSS INSIGHT",
        f"  • {TOSS_TIPS.get('field', {}).get(dew_key, 'Toss decision is contextual.')}",
        "",
        f" PHASE NOTE ({phase})",
        f"  • {PHASE_TIPS.get(phase, 'Focus on execution.')}",
        "",
        f" PLAYER FOCUS",
        f"  • {player_tip}",
        "",
        f"  DISCLAIMER: This strategy is AI-generated from statistical models.",
        f"   Final decisions rest with coaching staff.",
        "",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  KDAC-4 CricAI Platform",
        "=" * 60,
    ]

    return "\n".join(lines)


def generate_strategy_llm(prediction_result: dict, api_key: str = None) -> str:
    """
    Optional: Generate strategy using OpenAI/Gemini API.
    Falls back to template if no API key provided.
    """
    if not api_key:
        return generate_strategy(prediction_result)

    try:
        import openai
        openai.api_key = api_key
        prompt = f"""
You are a professional T20 cricket analyst. Based on the following ML prediction data,
generate a concise, actionable strategy brief (max 200 words) for the coaching staff.

Match: {prediction_result['team_a']} vs {prediction_result['team_b']}
Predicted Winner: {prediction_result['predicted_winner']}
Win Probability: {prediction_result['win_prob_a']}% vs {prediction_result['win_prob_b']}%
Confidence: {prediction_result['confidence_score']}/100
Key Factors: {', '.join(prediction_result.get('top_factors', []))}
Pitch: {prediction_result.get('pitch_type')}
Phase: {prediction_result.get('phase')}

Write strategy sections: Batting Plan, Bowling Plan, Key Watchout, and Toss Advice.
"""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.7
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"LLM API error: {e} — falling back to template.")
        return generate_strategy(prediction_result)


if __name__ == "__main__":
    # Demo with mock prediction result
    mock_result = {
        "team_a": "India",
        "team_b": "Australia",
        "predicted_winner": "India",
        "win_prob_a": 62.3,
        "win_prob_b": 37.7,
        "confidence_score": 74.6,
        "pitch_type": "spin-friendly",
        "phase": "Semi-Final",
        "top_factors": [
            "ICC Ranking advantage: India ranked #1",
            "Recent form: India win rate 78%",
            "Pitch condition: spin-friendly",
            "Toss advantage: Toss-winner elected to field"
        ]
    }
    print(generate_strategy(mock_result, dew_level="Medium"))
