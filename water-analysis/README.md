# 💧 Water Potability Predictor

Predicts whether water is **safe to drink** using a **Random Forest Classifier** trained on the real Kaggle Water Potability dataset (3,276 samples).

## Setup & Run

```bash
pip install -r requirements.txt
python model/train.py
streamlit run app/app.py
```

## Project Structure
```
water-analysis/
├── app/app.py                  # Streamlit web app
├── dataset/water_potability.csv  # Real dataset (3276 samples)
├── model/
│   ├── train.py                # Training script
│   ├── rf_model.pkl            # Trained classifier
│   └── scaler.pkl              # StandardScaler
├── utils/preprocess.py         # Preprocessing utilities
├── plots/                      # Training evaluation plots
└── requirements.txt
```

## Model Performance
| Metric     | Value |
|------------|-------|
| Accuracy   | 67%   |
| ROC-AUC    | 0.69  |
| CV ROC-AUC | 0.69 ± 0.02 |

> The dataset has inherent noise and class imbalance (61% not potable), which is typical for real-world water quality data.

## Potability Scale
| Probability | Verdict        |
|-------------|----------------|
| ≥ 70%       | Potable ✅      |
| 50–69%      | Likely Potable  |
| 35–49%      | Uncertain ⚠️   |
| < 35%       | Not Potable ❌  |
