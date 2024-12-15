import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Завантаження історичних даних акцій
symbol = "AAPL"  # Символ акцій компанії (можна замінити)
data = yf.download(symbol, start="2023-01-01", end="2024-01-01")

# Додавання частоти до індексу
data.index = pd.to_datetime(data.index)
data = data.asfreq('B')  # Частота 'B' для робочих днів

# Перевірка пропущених значень
print("Пропущені значення в даних:")
print(data.isnull().sum())

# Заповнення пропущених значень
data = data.interpolate()

# Графік зміни ціни закриття
data['Close'].plot(title=f"Ціна закриття {symbol}", figsize=(10, 6))
plt.xlabel("Дата")
plt.ylabel("Ціна закриття")
plt.tight_layout()
plt.savefig('practice-8/closing_price.png')
plt.close()

# Базова описова статистика
print("Базова описова статистика:")
print(data['Close'].describe())

# Декомпозиція часового ряду
result = seasonal_decompose(data['Close'], model='additive', period=30)
result.plot()
plt.tight_layout()
plt.savefig('practice-8/decomposition.png')
plt.close()

# Аналіз компонентів часового ряду
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Побудова ковзних середніх
data['SMA_7'] = data['Close'].rolling(window=7).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label="Ціна закриття")
plt.plot(data['SMA_7'], label="7-денне ковзне середнє")
plt.plot(data['SMA_30'], label="30-денне ковзне середнє")
plt.legend()
plt.title("Ковзні середні")
plt.tight_layout()
plt.savefig('practice-8/moving_averages.png')
plt.close()

# Розрахунок RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data['Close'])
plt.figure(figsize=(10, 6))
plt.plot(data['RSI'], label="RSI")
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title("Індекс відносної сили (RSI)")
plt.legend()
plt.tight_layout()
plt.savefig('practice-8/rsi.png')
plt.close()

# Волатильність
data['Volatility_30'] = data['Close'].rolling(window=30).std()
plt.figure(figsize=(10, 6))
plt.plot(data['Volatility_30'], label="30-денна волатильність")
plt.title("Волатильність")
plt.legend()
plt.tight_layout()
plt.savefig('practice-8/volatility.png')
plt.close()

# Прогнозування
train = data['Close'][:-30]
test = data['Close'][-30:]

# Експоненційне згладжування
model_es = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
fit_es = model_es.fit()
pred_es = fit_es.forecast(len(test))

# ARIMA
model_arima = ARIMA(train, order=(3, 1, 2))  # Змінено параметри ARIMA
fit_arima = model_arima.fit()
pred_arima = fit_arima.forecast(len(test))

# Оцінка якості прогнозу
mse_es = mean_squared_error(test, pred_es)
mae_es = mean_absolute_error(test, pred_es)

mse_arima = mean_squared_error(test, pred_arima)
mae_arima = mean_absolute_error(test, pred_arima)

print(f"Експоненційне згладжування - MSE: {mse_es}, MAE: {mae_es}")
print(f"ARIMA - MSE: {mse_arima}, MAE: {mae_arima}")

# Візуалізація прогнозів
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label="Реальні значення")
plt.plot(test.index, pred_es, label="Прогноз ES")
plt.plot(test.index, pred_arima, label="Прогноз ARIMA")
plt.legend()
plt.title("Прогнозування")
plt.tight_layout()
plt.savefig('practice-8/forecasting.png')
plt.close()
