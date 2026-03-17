# Titanic — Machine Learning with sklearn

Predicting survival of Titanic passengers. First project using sklearn after manually implementing ML algorithms from scratch.

## Results

| Model | Accuracy |
|---|---|
| Neural Network (MLP) | **84.92%** |
| Random Forest | 83.24% |
| Logistic Regression | 82.12% |
| Decision Tree | 81.56% |
| KNN | 80.45% |

**Kaggle score: 0.77272**

## Dataset

[Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) (Kaggle)

891 passengers, 12 columns. Task — predict survival (0/1).

## Data Processing

**Dropped:** `PassengerId`, `Name` (after feature engineering), `Ticket`, `Cabin` (77% missing), `Embarked`

**Missing value imputation:**
- `Age` — Decision Tree Regressor trained on `Pclass`, `Sex`, `Title`, `FamilySize` and other features (RMSE: 10.66 vs grouped median: 12.59)
- `Fare` — median

**Feature Engineering:**
- `Title` — extracted from name (Mr, Miss, Mrs, Master, Rare)
- `FamilySize` = SibSp + Parch + 1
- `IsAlone` — 1 if traveling without family
- `AgeBand` — age split into categories (0-12, 12-18, 18-35, 35-60, 60+)

**Encoding:**
- `Sex`: male=0, female=1
- `Title`: Mr=0, Miss=1, Mrs=2, Master=3, Rare=4

## Key Design Decisions

Different models use different feature sets and age imputation methods based on their nature:

| Model | Features | Age Imputation | Scaling |
|---|---|---|---|
| Neural Network | Full set | Decision Tree | ✅ |
| Random Forest | Full set | Decision Tree | ❌ |
| Decision Tree | Full set | Decision Tree | ❌ |
| Logistic Regression | Base features only | Grouped Median | ✅ |
| KNN | Base features only | Decision Tree | ✅ |

Logistic Regression and KNN use reduced features (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`) because correlated features like `FamilySize` hurt linear models and distance-based models.

## Project Structure

```
titanic-ml/
├── main.py          # model comparison on train/test split
├── submission.py    # generating predictions for Kaggle
├── train.csv        # training data
├── test.csv         # test data for Kaggle
└── submission.csv   # predictions for submission
```

## How to Run

```bash
pip install pandas numpy scikit-learn

# Compare models
python main.py

# Generate Kaggle submission
python submission.py
```

## Learning Context

This project is part of my ML learning path from scratch:

1. ✅ [ML-From-Scratch](https://github.com/Vitaliy-Pudlovskyy) — linear regression, logistic regression, XOR neural network manually in NumPy
2. ✅ [monobank-ml](https://github.com/Vitaliy-Pudlovskyy) — regression, classification, decision tree, KNN, random forest without any libraries
3. ✅ **titanic-ml** — first project with sklearn
4. 🔜 Customer Churn — XGBoost, LightGBM, CatBoost
5. 🔜 PyTorch — MLP, CNN, RNN/LSTM
