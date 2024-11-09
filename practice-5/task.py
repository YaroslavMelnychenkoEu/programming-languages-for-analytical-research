import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from pathlib import Path

# Завантаження даних
df = pd.read_csv(Path(__file__).parent / "Mall_Customers.csv")

# Огляд даних
print(df.head())

# Частина 1. Підготовка даних

# Перевірка на пропущені значення
print(df.isnull().sum())

# Описова статистика
print(df.describe())

# Гістограми для кожної змінної
plt.figure(figsize=(15, 5))
for i, col in enumerate(['Age', 'Annual_Income', 'Spending_Score'], start=1):
    plt.subplot(1, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Розподіл {col}')
plt.savefig("practice-5/histograms.png")
plt.close()

# Стандартизація змінних для кластеризації
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'Annual_Income', 'Spending_Score']])

# Частина 2. Визначення оптимальної кількості кластерів

# Метод ліктя (Elbow method)

# Обчислення інерції для різних значень k
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Побудова графіку для методу ліктя
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Кількість кластерів k')
plt.ylabel('Інерція')
plt.title('Метод ліктя для вибору оптимального k')
plt.savefig("practice-5/elbow_method.png")
plt.close()

# Розрахунок коефіцієнта силуету для різних значень k
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(df_scaled)
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Побудова графіку коефіцієнта силуету
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, 'bo-')
plt.xlabel('Кількість кластерів k')
plt.ylabel('Коефіцієнт силуету')
plt.title('Silhouette Score для вибору оптимального k')
plt.savefig("practice-5/silhouette_score.png")
plt.close()

# Частина 3. Кластеризація та аналіз результатів

# Виконання кластеризації методом K-means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_

# Розмасштабування центроїдів до вихідних значень
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Візуалізація результатів кластеризації
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score', hue='Cluster', palette='viridis', s=60)
plt.scatter(centroids[:, 1], centroids[:, 2], s=100, c='red', label='Центроїди')
plt.legend()
plt.title('Кластери клієнтів методом K-means')
plt.xlabel('Річний дохід')
plt.ylabel('Показник витрат')
plt.savefig("practice-5/kmeans_clusters.png")
plt.close()

# Обчислення середніх значень тільки для числових показників для кожного кластера
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print(cluster_summary)

# Частина 4. Додаткові завдання

# Виконання кластеризації методом DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)

# Візуалізація результатів DBSCAN
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score', hue='DBSCAN_Cluster', palette='viridis', s=60)
plt.title('Кластери клієнтів методом DBSCAN')
plt.xlabel('Річний дохід')
plt.ylabel('Показник витрат')
plt.savefig("practice-5/dbscan_clusters.png")
plt.close()