# CSE 4820 Final Project — Machine Learning on Gaming Habits & Diabetes Risk

Two end-to-end ML projects: **(1)** predicting a student's final grade from their gaming/lifestyle habits (regression), and **(2)** classifying diabetes from clinical measurements (classification). Full write-up in [`CSE 4820 Final Report.pdf`](CSE%204820%20Final%20Report.pdf).

---

## Project 1 — Can a "gaming lifestyle" predict grades? (Regression)

### Problem
Given ~6,400 students with features like gaming hours, study hours, sleep, attendance, device usage, reaction time, addiction score, and stress level — predict their final grade (0–100).

### Approach
- **EDA first** (`CSE4820_FinalProj.ipynb`): correlation heatmaps, KDE plots, and agglomerative clustering showed study hours and stress level are strong predictors, reaction time is tied to device usage/addiction, and age/social activity/gaming genre carry little signal.
- **Model search** (`CSE4820_FinalProjModel.ipynb`): compared SGD regression with polynomial features, SVR (RBF), Kernel Ridge, KNN, Random Forest, MLP, a Nystroem kernel-approximation pipeline, and HistGradientBoosting — all tuned with cross-validated grid/randomized search.
- Prediction correlations across models were ~0.99, meaning every model learned the same signal and stacking wouldn't help — so the simplest top performer, **HistGradientBoosting**, was chosen as the final model.

### Results
| Final model | Val MAE | Val RMSE | Val R² |
|---|---|---|---|
| HistGradientBoosting | **4.49 points** | 5.77 | **0.935** |

Predictions land within ~4.5 grade points on average. Mapped to letter grades, the model gets the exact letter right ~73% of the time, is essentially always within one letter grade, and almost never misses a failing student. **Takeaway: gaming lifestyle is a genuinely strong predictor of academic performance.**

![Predicted vs actual grade](figures/predicted_vs_actual_grade.png)

---

## Project 2 — Diabetes screening from clinical data (Classification)

### Problem
Classify whether a patient has diabetes using 8 clinical measurements (Pima Indians dataset: 768 patients, ~35% positive).

### Approach (`CSE4820_DiabetesProj.ipynb`)
- **Key data issue:** zeros in Insulin (374/768) and SkinThickness (227/768) are really missing values — converted to NaN and median-imputed.
- Added interaction features (BMI², BMI×Glucose) since the glucose–BMI relationship drives this dataset.
- Compared **Logistic Regression (L1 + polynomial features), SVC (RBF), and HistGradientBoosting**, tuned by cross-validated search and compared on ROC-AUC.

### Results
| Model | Val AUC | Val Accuracy | Val F1 |
|---|---|---|---|
| **SVC (RBF)** | **0.873** | **0.805** | 0.692 |
| Logistic Regression | 0.871 | 0.789 | 0.675 |
| HistGradientBoosting | 0.856 | 0.781 | 0.682 |

Held-out test: ~72% accuracy, AUC ≈ 0.82. Glucose dominates feature importance; Insulin/SkinThickness contribute little due to missingness. For a real screening tool, lowering the decision threshold from 0.5 to ~0.35–0.40 trades precision for the recall needed to avoid missing diabetic patients.

![ROC curves](figures/diabetes_roc_curves.png)

---

## Repository guide

| File | Contents |
|---|---|
| `CSE4820_FinalProj.ipynb` | Data exploration — gaming/grades regression problem |
| `CSE4820_FinalProjModel.ipynb` | Model training & evaluation — gaming/grades regression problem |
| `CSE4820_DiabetesProj.ipynb` | Full pipeline (EDA + models) — diabetes classification problem |
| `CSE 4820 Final Report.pdf` | Written final report |

Code is split across three notebooks to keep any single one from getting overwhelmingly long.

## How to run
Open any notebook in VS Code or Colab and hit **Run All**. Place `Gaming_Academic_Performance.csv` and `diabetes.csv` in the same directory as the notebooks. All figures (including the ones embedded above) are saved automatically to a `figures/` folder — commit that folder after a run so the images render on GitHub.

**Requirements:** `scikit-learn 1.8.0` · `pandas 3.0.2` · `numpy 2.4.4` · `matplotlib 3.10.9` · `seaborn 0.13.2`
