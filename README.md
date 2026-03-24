# 🚨 Credit Card Fraud Detection API

End-to-end ML system detecting fraudulent transactions (**ROC-AUC 0.95+** on the Kaggle creditcard dataset with the bundled training pipeline).

## Quickstart

```bash
pip install -r requirements.txt
python -m src.train
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health:** [http://localhost:8000/health](http://localhost:8000/health)

Train from repo root with `python -m src.train` so imports resolve (`src` package). After training, the API loads `models/preprocessing_pipeline.pkl` and `models/best_model.pkl`.

### Dataset (keeps the Git repo small)

`data/raw/creditcard.csv` is **not committed** (often ~150MB+). Download it from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/raw/`. Trained `models/*.pkl` files are also **gitignored**—run `python -m src.train` after cloning.

## Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" fraud-api
```

On Windows (PowerShell), mount models with:

```powershell
docker run -p 8000:8000 -v "${PWD}/models:/app/models" fraud-api
```

The image expects trained artifacts under `/app/models` (use the volume so you do not bake secrets or large binaries into the image).

## Tech

- **ML:** scikit-learn, XGBoost
- **API:** FastAPI, Pydantic
- **Deployment:** Docker

## Project layout (summary)

```text
api/          FastAPI app
src/train.py  Training pipeline (LogisticRegression vs RandomForest, best by ROC-AUC)
models/       Saved `.pkl` artifacts (joblib)
data/raw/     creditcard.csv (not required inside Docker if you mount `models/` only)
```

## Expected metrics

Highly imbalanced fraud data: prefer ROC-AUC / PR-AUC / recall–precision trade-offs over accuracy alone. Typical strong baselines on this dataset land around **ROC-AUC ≥ 0.95** with threshold tuned to business cost.
