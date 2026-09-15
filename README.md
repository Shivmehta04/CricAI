# 🏏 CricAI — ICC T20 World Cup 2026 Match Outcome Prediction Platform

**KDAC-4 | GLS Nexus Hackathon 2026 | Kenex AI**

An end-to-end data analytics & ML platform that predicts T20 cricket match outcomes and generates actionable coaching strategy notes using GenAI.

---

# CricAI Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cric-ai.streamlit.app)

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (simulate → ETL → train → predict → strategy)
python run_demo.py

# 3. Launch dashboard
streamlit run src/dashboard/app.py
```

---

## 📁 Project Structure

```
kdac4-cricket-prediction/
├── data/
│   ├── raw/                    # Simulated CSV datasets
│   ├── processed/              # ETL-cleaned datasets
│   ├── models/                 # Trained model artifacts (.pkl)
│   └── cricket_warehouse.db   # SQLite data warehouse
├── src/
│   ├── simulation/
│   │   └── generate_data.py   # Synthetic dataset generator
│   ├── etl/
│   │   └── pipeline.py        # Extract → Transform → Load
│   ├── models/
│   │   ├── train_model.py     # XGBoost + RF training
│   │   └── predict.py         # Win probability prediction engine
│   ├── dashboard/
│   │   └── app.py             # Streamlit EDA + predictor UI
│   ├── genai/
│   │   └── strategy.py        # GenAI strategy note generator
│   └── api/
│       └── main.py            # FastAPI REST backend
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── run_demo.py                 # One-click full pipeline
└── README.md
```

---

## 🧩 Platform Components

| Component | Technology | Purpose |
|---|---|---|
| Data Simulation | Python / NumPy | Generate 500+ realistic T20 match records |
| ETL Pipeline | Pandas / SQLite | Clean, transform, engineer features |
| Data Warehouse | SQLite | Structured storage with 6 tables |
| EDA Dashboard | Streamlit + Plotly | Interactive team/venue/toss analysis |
| ML Model | XGBoost + Random Forest | Win probability prediction |
| Strategy Layer | Template NLG / OpenAI | Plain-language coaching recommendations |
| REST API | FastAPI | `/predict`, `/strategy`, `/teams` endpoints |
| Deployment | Docker + Docker Compose | Containerized 3-service deployment |

---

## 📊 ML Model Details

- **Algorithm**: XGBoost Classifier (fallback: Gradient Boosting)
- **Features**: 17 features including ICC ranking, win rate, pitch type, dew factor, toss decision, match phase
- **Validation**: 5-fold Stratified Cross-Validation
- **Metrics**: Accuracy ~70–75%, AUC-ROC ~0.72–0.78

### Key Features Used:
1. ICC Ranking differential
2. Win rate differential (last 2 years)
3. Average team score
4. Net Run Rate (NRR)
5. Pitch type (spin/pace/batting/swing)
6. Toss decision & winner
7. Dew factor
8. Tournament phase
9. Weather conditions (temperature, humidity)

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/teams` | List all teams |
| GET | `/teams/{name}` | Team stats |
| GET | `/matches/recent` | Recent match results |
| POST | `/predict` | Predict match outcome |
| POST | `/strategy` | Generate strategy brief |
| POST | `/predict-and-strategy` | Combined one-call endpoint |

API Docs: http://localhost:8000/docs

---

## 🐳 Docker Deployment

```bash
# Build and run all services
docker-compose -f docker/docker-compose.yml up --build

# Services:
# - Dashboard : http://localhost:8501
# - API       : http://localhost:8000
# - API Docs  : http://localhost:8000/docs
```

---

## 🔮 GenAI Strategy Layer

The GenAI layer converts ML prediction output into human-readable coaching briefs covering:
- **Batting Strategy** (pitch-specific)
- **Bowling Plan** (containment + attack)
- **Toss Advice** (dew-aware)
- **Phase-specific tips** (Group Stage vs Final)
- **Key Player Roles**

Optional LLM integration: plug in your OpenAI or Gemini API key in `.env`.

---

## 📈 USP — What Makes CricAI Different

| Feature | CricAI | ESPNcricinfo Analytics | Traditional Scouting |
|---|---|---|---|
| Real-time prediction | ✅ | ❌ | ❌ |
| Venue + weather integration | ✅ | Partial | ❌ |
| GenAI strategy notes | ✅ | ❌ | ❌ |
| Open API | ✅ | ❌ | ❌ |
| Self-updating warehouse | ✅ | ❌ | ❌ |
| Dockerized deployment | ✅ | N/A | N/A |

---

## 👥 Team

**Kenex AI | GLS Nexus Hackathon 2026**
Problem Statement: KDAC-4 — ICC Men's T20 Cricket Match World Cup 2026 Outcome Prediction

---

## 📄 License

MIT License — For academic/hackathon use only.
