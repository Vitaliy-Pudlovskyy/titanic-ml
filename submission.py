import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ============================================================
# ЗАВАНТАЖЕННЯ
# ============================================================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test_ids = test['PassengerId']

# ============================================================
# ФУНКЦІЯ ОБРОБКИ — однакова для train і test
# ============================================================
def preprocess(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    df['Title'] = df['Title'].replace(
        ['Dr', 'Rev', 'Major', 'Mlle', 'Col', 'Don', 'Mme', 'Ms',
         'Lady', 'Sir', 'Capt', 'the Countess', 'Jonkheer'], 'Rare'
    )
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    return df

train = preprocess(train)
test = preprocess(test)

# ============================================================
# ЗАПОВНЕННЯ ВІКУ
# ============================================================
age_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Title', 'FamilySize']

dt_age = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_age.fit(train[train['Age'].notnull()][age_features],
           train[train['Age'].notnull()]['Age'])

train.loc[train['Age'].isnull(), 'Age'] = dt_age.predict(
    train[train['Age'].isnull()][age_features])
test.loc[test['Age'].isnull(), 'Age'] = dt_age.predict(
    test[test['Age'].isnull()][age_features])

train['AgeBand'] = pd.cut(train['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4])
test['AgeBand'] = pd.cut(test['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4])

# ============================================================
# НАВЧАННЯ І ПЕРЕДБАЧЕННЯ
# ============================================================
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Title', 'FamilySize', 'IsAlone', 'AgeBand']

X_train = train[features].fillna(0)
X_test = test[features].fillna(0)
y_train = train['Survived']


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Ensemble — голосування більшості
nn_pred = model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

# Якщо обидві моделі згодні — беремо їх результат
# Якщо ні — перевага NN (вона краща)
predictions = nn_pred

# ============================================================
# ЗБЕРЕЖЕННЯ
# ============================================================
submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': predictions})
submission.to_csv('submission.csv', index=False)
print('submission.csv збережено!')
print(f'Передбачень: {len(submission)}')