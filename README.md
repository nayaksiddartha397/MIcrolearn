# LoanSight — Loan Repayment Predictor

Predicts whether a Lending Club loan will be repaid or default.
**Zero sklearn** — every model, scaler, encoder, imputer, and grid search
is powered by [microlearn](./microlearn/), a NumPy-only ML library.

---

## Project structure

```
loan-predictor/
├── microlearn/          ← your library (copy here — see Step 3)
├── data/                ← put the Lending Club CSV here (gitignored)
├── model/               ← auto-created by train.py; holds model.pkl
├── static/
│   └── index.html       ← React single-page app (no build step)
├── train.py             ← feature engineering + GridSearchCV + save
├── app.py               ← Flask REST API (/api/predict, /api/metrics)
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .dockerignore
└── README.md
```

---

## Dataset — download first

1. Go to **https://www.kaggle.com/datasets/wordsforthewise/lending-club**
2. Download `accepted_2007_to_2018Q4.csv.gz`
3. Put it inside the `data/` folder (no need to unzip — pandas reads `.gz` directly)

The file is ~1.7 GB unzipped. Training uses up to 200,000 rows by default
(adjustable via `MAX_ROWS` in `train.py`).

---

## Setup & run — macOS

```bash
# 1. Enter the project folder
cd loan-predictor

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Copy your microlearn package into the project root
#    (the folder that contains __init__.py, tree.py, preprocessing.py, etc.)
cp -r /path/to/your/microlearn ./microlearn

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train the model
#    — quick synthetic demo (no CSV needed, ~2 min):
python train.py

#    — real Lending Club data (~5-15 min depending on machine):
python train.py data/accepted_2007_to_2018Q4.csv.gz

# 6. Start the server
python app.py

# 7. Open the app
open http://localhost:5000
```

---

## Setup & run — Windows

```bat
REM 1. Enter the project folder
cd loan-predictor

REM 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

REM 3. Copy your microlearn package
xcopy /E /I C:\path\to\your\microlearn microlearn

REM 4. Install dependencies
pip install -r requirements.txt

REM 5. Train the model (synthetic demo)
python train.py

REM    Or with real data:
python train.py data\accepted_2007_to_2018Q4.csv.gz

REM 6. Start the server
python app.py

REM 7. Open http://localhost:5000 in your browser
```

---

## Docker

```bash
# Build (trains synthetic model at build time, ~3 min)
docker build -t loansight .

# Run
docker run -p 5000:5000 loansight

# Push to Docker Hub
docker login
docker tag loansight YOUR_USERNAME/loansight:latest
docker push YOUR_USERNAME/loansight:latest
```

**Faster builds after first train:** uncomment `COPY model/ ./model/` in the
Dockerfile and comment out `RUN python train.py` — this reuses your local
`model.pkl` and cuts build time from ~3 min to ~10 seconds.

---

## Push to GitHub

```bash
cd loan-predictor
git init
git add .
git commit -m "feat: loan repayment predictor — microlearn RandomForest, no sklearn"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/loan-predictor.git
git push -u origin main
```

---

## API reference

### `POST /api/predict`

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000, "int_rate": 13.5, "installment": 455,
    "term": "36 months", "purpose": "debt_consolidation",
    "annual_inc": 65000, "dti": 18,
    "emp_length": "5 years", "home_ownership": "RENT",
    "verification_status": "Verified",
    "revol_bal": 8500, "revol_util": 0.45,
    "open_acc": 8, "total_acc": 22,
    "delinq_2yrs": 0, "inq_last_6mths": 1,
    "mths_since_last_delinq": 0, "pub_rec": 0,
    "years_since_earliest_cr_line": 8,
    "years_since_last_credit_pull": 1
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.2341,
  "verdict": "APPROVED",
  "risk_level": "low",
  "risk_color": "#22c55e",
  "top_factors": [
    { "feature": "dti", "importance": 0.148, "value": 18.0 },
    ...
  ]
}
```

### `GET /api/metrics`

Returns test-set accuracy, ROC-AUC, confusion matrix, classification
report, and top-10 feature importances.

---

## microlearn components used

| File       | Components |
|------------|-----------|
| `train.py` | `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `Pipeline`, `ColumnTransformer`, `RandomForestClassifier`, `GridSearchCV`, `train_test_split`, `metrics.*` |
| `app.py`   | Loads the fitted `Pipeline` via pickle — no extra microlearn calls at inference time |

---

## Expected results (real Lending Club data)

| Metric   | Typical range |
|----------|--------------|
| Accuracy | 0.82 – 0.87  |
| ROC-AUC  | 0.86 – 0.92  |
| Default rate | ~20 %    |

Top features by importance: `loan_amnt_to_annual_inc`, `dti`,
`int_rate`, `revol_util`, `log_annual_inc`.