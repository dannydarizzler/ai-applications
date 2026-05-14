---
title: Apartment Price Zurich
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
---

# Model Iterations Documentation
## Task: Apartment Price Prediction (Regression)

---

## Summary of Iterative Process

| Iteration | Objective | Key Changes | Models Used | CV Mean R² | CV Std Dev | Change in Performance | Fit Diagnosis |
|-----------|-----------|-------------|-------------|------------|------------|-----------------------|---------------|
| **1** | Build baseline model | - Drop missing values<br>- Remove duplicates<br>- StandardScaler<br>- 5-fold CV | Linear Regression<br>Random Forest (n=100) | 0.4739 (LR)<br>0.4648 (RF) | 0.0980 (LR)<br>0.1275 (RF) | Baseline | ☐ Overfitting ☑ Underfitting ☐ Good Fit |
| **2** | Improve with new feature & better models | - All steps from Iter 1<br>- New feature: `area_per_room`<br>- 5-fold CV | Ridge (alpha=10)<br>Gradient Boosting (n=200, depth=4, lr=0.05) | 0.4786 (Ridge)<br>0.4977 (GB) | 0.1099 (Ridge)<br>0.1005 (GB) | +0.0329 improvement | ☐ Overfitting ☐ Underfitting ☑ Good Fit |

---

## New Feature

**`area_per_room`** = living area (m²) / number of rooms

- Represents the average room size of the apartment
- Not included in prior exercises (Week 1 and Week 2 used `area` and `rooms` separately)
- Motivation: a 3-room apartment with 90 m² (30 m²/room) is worth more than one with 60 m² (20 m²/room), even with the same room count

---

## Preprocessing Steps

1. Load CSV data (`original_apartment_data_analytics_hs24.csv`)
2. Drop rows with missing values (`dropna()`)
3. Remove duplicate rows (`drop_duplicates()`)
4. Engineer new feature: `area_per_room = area / rooms`
5. Apply `StandardScaler` to all features (inside sklearn Pipeline)
6. Evaluate with 5-fold cross-validation (R² metric)

---

## Models Used

| Model | Iteration | Hyperparameters | Notes |
|-------|-----------|-----------------|-------|
| Linear Regression | 1 | default | Baseline, prone to underfitting |
| Random Forest | 1 | n_estimators=100, random_state=42 | Overfits without tuning |
| Ridge Regression | 2 | alpha=10.0 | L2 regularization reduces overfitting |
| Gradient Boosting | 2 | n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42 | Best overall performer |

---

## Evaluation Method

- **Metric:** R² (coefficient of determination)
- **Validation strategy:** 5-fold cross-validation on the full dataset
- Higher R² = better fit (1.0 = perfect, 0.0 = predicts mean, <0 = worse than mean)

---

## Final Selected Model

**Gradient Boosting Regressor** (Iteration 2)

**Reason for selection:**
- Highest CV R² across all models and iterations
- Handles non-linear relationships between features and price
- Lower overfitting compared to Random Forest thanks to shallow trees (max_depth=4) and slow learning rate (0.05)
- Benefits from the new `area_per_room` feature

**Final features used (8 total):**
- `rooms` – number of rooms
- `area` – living area in m²
- `pop` – municipality population
- `pop_dens` – population density (per km²)
- `frg_pct` – percentage of foreign residents
- `emp` – number of employees in municipality
- `tax_income` – average taxable income in municipality
- `area_per_room` – **new feature**: average room size (area / rooms)

---

## Application

The trained model is served via a Gradio web interface (`app.py`).
Users select the number of rooms, living area, and municipality — the app returns the predicted monthly rent in CHF.
