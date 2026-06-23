# FraudShield AI — Deployment Guide

## Local Development

```bash
git clone https://github.com/your-username/fraudshield-ai.git
cd fraudshield-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models (generates data + saves model artifacts)
python models/train.py

# Launch dashboard
streamlit run app.py
# → Open http://localhost:8501
```

---

## Docker

```bash
# Build
docker build -t fraudshield-ai .

# Run
docker run -p 8501:8501 fraudshield-ai

# With persistent data volume
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  fraudshield-ai
```

---

## Render (Free Tier)

1. Push project to GitHub (ensure `models/` is not in `.gitignore` or pre-train and commit artifacts).
2. Go to [render.com](https://render.com) → New Web Service.
3. Connect your GitHub repo.
4. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt && python models/train.py`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
5. Click **Deploy**.

---

## Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Railway auto-detects `Procfile`.

---

## Heroku

```bash
# Login
heroku login

# Create app
heroku create fraudshield-ai

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FRAUDSHIELD_DB` | `data/fraudshield.db` | SQLite database path |
| `PORT` | `8501` | Streamlit server port (set automatically by cloud platforms) |

---

## Pre-training Models Before Deployment

Because cloud environments may not have enough compute time on startup,
it's recommended to train locally and commit the model artifacts:

```bash
python models/train.py
git add models/best_model.pkl models/preprocessor.pkl models/model_metrics.json
git commit -m "Add trained model artifacts"
git push
```

Then remove these from `.gitignore` for the deployment repository.
