import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

# 1. Загрузка данных 
# df = pd.read_csv('data.csv')

# Демонастрационные данные
import numpy as np
np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 5)  # 5 признаков
y = np.random.randint(0, 2, n_samples)  # 0 или 1

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Обучение модели
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Оценка качества
y_pred = model.predict(X_test)
print("Метрики модели:")
print(classification_report(y_test, y_pred))

# 4. Сохранение модели
os.makedirs('models', exist_ok=True)
with open('models/model_v1.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Модель успешно сохранена в models/model_v1.pkl")
