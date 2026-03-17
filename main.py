import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('train.csv')

# ============================================================
# FEATURE ENGINEERING
# ============================================================
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
df['Title'] = df['Title'].replace(
    ['Dr', 'Rev', 'Major', 'Mlle', 'Col', 'Don', 'Mme', 'Ms',
     'Lady', 'Sir', 'Capt', 'the Countess', 'Jonkheer'], 'Rare'
)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# ============================================================
# КОДУВАННЯ
# ============================================================
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})

# ============================================================
# ЗАПОВНЕННЯ ВІКУ
# ============================================================
df_known = df[df['Age'].notnull()].copy()
df_missing = df[df['Age'].isnull()].copy()

age_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Title', 'FamilySize']
X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(
    df_known[age_features], df_known['Age'], test_size=0.2, random_state=42
)

grouped_median = df_known.groupby(['Pclass', 'Sex'])['Age'].median()
rmse_median = np.sqrt(mean_squared_error(y_age_test,
    X_age_test.apply(lambda r: grouped_median[r['Pclass'], r['Sex']], axis=1)))

dt_age = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_age.fit(X_age_train, y_age_train)
rmse_dt = np.sqrt(mean_squared_error(y_age_test, dt_age.predict(X_age_test)))

rf_age = RandomForestRegressor(n_estimators=100, random_state=42)
rf_age.fit(X_age_train, y_age_train)
rmse_rf = np.sqrt(mean_squared_error(y_age_test, rf_age.predict(X_age_test)))



# Версія 1: Decision Tree — для RF і DT
df_dt = df.copy()
df_dt.loc[df_dt['Age'].isnull(), 'Age'] = dt_age.predict(df_missing[age_features])
df_dt['AgeBand'] = pd.cut(df_dt['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])

# Версія 2: Grouped Median — для LogReg і KNN
df_med = df.copy()
df_med['Age'] = df_med.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
df_med['AgeBand'] = pd.cut(df_med['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])



# ============================================================
# ПІДГОТОВКА ДО МОДЕЛЕЙ
# ============================================================
df_dt = df_dt.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
df_med = df_med.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])

y = df_dt['Survived']

X_full = df_dt.drop(columns=['Survived'])
X_lr = df_med[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
X_knn = df_dt[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

X_full_train, X_full_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
X_lr_train, X_lr_test, _, _ = train_test_split(X_lr, y, test_size=0.2, random_state=42)
X_knn_train, X_knn_test, _, _ = train_test_split(X_knn, y, test_size=0.2, random_state=42)

scaler_lr = StandardScaler()
X_lr_train_scaled = scaler_lr.fit_transform(X_lr_train)
X_lr_test_scaled = scaler_lr.transform(X_lr_test)

scaler_knn = StandardScaler()
X_knn_train_scaled = scaler_knn.fit_transform(X_knn_train)
X_knn_test_scaled = scaler_knn.transform(X_knn_test)

# ============================================================
# МОДЕЛІ
# ============================================================

from sklearn.neural_network import MLPClassifier

scaler_nn = StandardScaler()
X_nn_train_scaled = scaler_nn.fit_transform(X_full_train)
X_nn_test_scaled = scaler_nn.transform(X_full_test)

nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42)
nn_model.fit(X_nn_train_scaled, y_train)
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000).fit(X_lr_train_scaled, y_train), X_lr_test_scaled),
    'Neural Network': (nn_model, X_nn_test_scaled),
    'Decision Tree':       (DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_full_train, y_train), X_full_test),
    'Random Forest':       (RandomForestClassifier(random_state=42).fit(X_full_train, y_train), X_full_test),
    'KNN':                 (KNeighborsClassifier().fit(X_knn_train_scaled, y_train), X_knn_test_scaled),
}


print('\n--- Результати v2 (з feature engineering) ---')
for name, (model, X_eval) in models.items():
    print(f'{name}: {accuracy_score(y_test, model.predict(X_eval)):.4f}')

