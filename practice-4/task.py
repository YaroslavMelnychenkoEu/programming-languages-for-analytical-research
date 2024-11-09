import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження даних
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

# додаємо цільову змінну (ціни на будинки) як новий стовпчик MedHouseVal
df['MedHouseVal'] = data.target

# Частина 1. Дослідницький аналіз даних

# Провести базовий аналіз даних:

# 1. Виведення описової статистики, виводить базову статистику (середнє, стандартне відхилення, мінімум, максимум тощо) для кожної ознаки
print(df.describe())

# 2. Перевірка на пропущені значення. Перевіряє, чи є пропущені значення в кожному стовпчику.
print(df.isnull().sum())

# 3. Визначення типів даних, виводить типи даних кожного стовпця
print(df.dtypes)

# Виконати візуальний аналіз:

# 1. Гістограми
df.hist(bins=30, figsize=(15, 10))
plt.savefig('practice-4/histograms.png')
plt.close()

# 2. Boxplot для виявлення викидів
plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.savefig('practice-4/boxplot.png')
plt.close()

# 3. Кореляційна матриця і теплова карта
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.savefig('practice-4/correlation_heatmap.png')
plt.close()

# 4. Scatter plots між ознаками і цільовою змінною
for feature in data.feature_names:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature], df['MedHouseVal'], alpha=0.3)
    plt.xlabel(feature)
    plt.ylabel('MedHouseVal')
    plt.savefig(f'practice-4/scatter_{feature}.png')
    plt.close()

# Частина 2. Підготовка даних

# Розділення на тренувальну і тестову вибірки (80/20)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Частина 3. Побудова моделей

# Проста лінійна регресія

# Використання ознаки з найвищою кореляцією з цільовою змінною
X_train_simple = X_train_scaled[:, [np.argmax(abs(corr_matrix['MedHouseVal'][:-1]))]]
X_test_simple = X_test_scaled[:, [np.argmax(abs(corr_matrix['MedHouseVal'][:-1]))]]

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)
y_pred_simple = simple_model.predict(X_test_simple)

# Оцінка моделі
mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)
print("Simple Linear Regression - MSE:", mse_simple, "R2:", r2_simple)

# Множинна лінійна регресія

multiple_model = LinearRegression()
multiple_model.fit(X_train_scaled, y_train)
y_pred_multiple = multiple_model.predict(X_test_scaled)

# Оцінка моделі
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)
print("Multiple Linear Regression - MSE:", mse_multiple, "R2:", r2_multiple)

# Оптимізована модель з регуляризацією

# Ridge регресія (L2 регуляризація)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Lasso регресія (L1 регуляризація)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

# Оцінка оптимізованих моделей
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_ridge = r2_score(y_test, y_pred_ridge)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("Ridge Regression - MSE:", mse_ridge, "R2:", r2_ridge)
print("Lasso Regression - MSE:", mse_lasso, "R2:", r2_lasso)

# Частина 4. Оцінка моделей

# Функція для обчислення всіх метрик
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - X_test.shape[1] - 1)
    print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}, Adjusted R2: {adj_r2}")

# Оцінка всіх моделей
print("Simple Linear Regression:")
evaluate_model(y_test, y_pred_simple)

print("Multiple Linear Regression:")
evaluate_model(y_test, y_pred_multiple)

print("Ridge Regression:")
evaluate_model(y_test, y_pred_ridge)

print("Lasso Regression:")
evaluate_model(y_test, y_pred_lasso)

# Візуалізація передбачених значень проти реальних значень для Ridge Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ridge, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Реальні значення")
plt.ylabel("Передбачені значення")
plt.title("Передбачені vs Реальні значення (Ridge Regression)")
plt.savefig("practice-4/predicted_vs_actual_ridge.png")
plt.close()

# Графік залишків для Ridge Regression
residuals = y_test - y_pred_ridge
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_ridge, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Передбачені значення")
plt.ylabel("Залишки")
plt.title("Графік залишків (Ridge Regression)")
plt.savefig("practice-4/residuals_plot_ridge.png")
plt.close()

# Розподіл залишків для Ridge Regression
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel("Залишки")
plt.title("Розподіл залишків (Ridge Regression)")
plt.savefig("practice-4/residuals_distribution_ridge.png")
plt.close()

def predict_price(features_dict, model, scaler):
    """
    Прогнозує ціну на будинок за заданими характеристиками.

    Parameters:
    - features_dict (dict): Словник з характеристиками будинку. 
      Ключі повинні відповідати назвам ознак, а значення — значенням ознак.
      Очікуються такі ключі:
      'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'.
    - model: Навчена модель для передбачення (Ridge Regression).
    - scaler: Об'єкт StandardScaler, використаний для масштабування ознак під час тренування.

    Returns:
    - float: Прогнозована ціна на будинок.
    """
    # Перетворюємо словник на список ознак у відповідному порядку
    feature_order = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    features = [features_dict[feature] for feature in feature_order]
    
    # Масштабуємо вхідні дані за допомогою збереженого скейлера
    features_scaled = scaler.transform([features])
    
    # Передбачаємо ціну за допомогою моделі
    predicted_price = model.predict(features_scaled)
    
    return predicted_price[0]

# Приклад використання функції:
sample_features = {
    'MedInc': 8.3252,
    'HouseAge': 41,
    'AveRooms': 7,
    'AveBedrms': 2,
    'Population': 500,
    'AveOccup': 3,
    'Latitude': 37.88,
    'Longitude': -122.23
}

predicted_price = predict_price(sample_features, ridge_model, scaler)
print(f"Прогнозована ціна на будинок: {predicted_price}")