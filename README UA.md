# Titanic — Machine Learning with sklearn

Передбачення виживання пасажирів Титаніка. Перший проект з використанням sklearn після ручної реалізації алгоритмів.

## Результати

| Модель | Точність |
|---|---|
| Neural Network (MLP) | **84.92%** |
| Random Forest | 83.24% |
| Logistic Regression | 82.12% |
| Decision Tree | 81.56% |
| KNN | 80.45% |

**Kaggle score: 0.77272**

## Датасет

[Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) (Kaggle)

891 пасажирів, 12 колонок. Задача — передбачити виживання (0/1).

## Обробка даних

**Видалено:** `PassengerId`, `Name` (після feature engineering), `Ticket`, `Cabin` (77% пропусків), `Embarked`

**Заповнення пропусків:**
- `Age` — Decision Tree Regressor на основі `Pclass`, `Sex`, `Title`, `FamilySize` та інших фіч (RMSE: 10.66 vs grouped median: 12.59)
- `Fare` — медіана

**Feature Engineering:**
- `Title` — витягнуто з імені (Mr, Miss, Mrs, Master, Rare)
- `FamilySize` = SibSp + Parch + 1
- `IsAlone` — 1 якщо подорожував без сім'ї
- `AgeBand` — вік розбитий на категорії (0-12, 12-18, 18-35, 35-60, 60+)

**Кодування:**
- `Sex`: male=0, female=1
- `Title`: Mr=0, Miss=1, Mrs=2, Master=3, Rare=4

## Особливості

Різні моделі використовують різні набори фіч і методи заповнення віку:

- **Neural Network, Random Forest, Decision Tree** — повний набір фіч + вік через Decision Tree
- **Logistic Regression** — базові фічі (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`) + вік через grouped median
- **KNN** — базові фічі + вік через Decision Tree + StandardScaler

## Структура проекту

```
titanic-ml/
├── main.py          # порівняння моделей на train/test split
├── submission.py    # генерація передбачень для Kaggle
├── train.csv        # тренувальні дані
├── test.csv         # тестові дані для Kaggle
└── submission.csv   # передбачення для сабмішну
```

## Запуск

```bash
pip install pandas numpy scikit-learn

# Порівняння моделей
python main.py

# Генерація submission для Kaggle
python submission.py
```

## Контекст навчання

Цей проект — частина шляху вивчення ML з нуля:

1. ✅ [ML-From-Scratch](https://github.com/Vitaliy-Pudlovskyy) — лінійна регресія, логістична регресія, XOR нейронна мережа вручну на NumPy
2. ✅ [monobank-ml](https://github.com/Vitaliy-Pudlovskyy) — regression, classification, decision tree, KNN, random forest вручну без бібліотек
3. ✅ **titanic-ml** — перший проект зі sklearn
4. 🔜 Customer Churn — XGBoost, LightGBM, CatBoost
5. 🔜 PyTorch — MLP, CNN, RNN/LSTM
