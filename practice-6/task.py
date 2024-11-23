# a) Завантаження датасету
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

df = pd.read_csv(Path(__file__).parent / "Mall_Customers.csv")

# b) Проведення первинного аналізу даних (EDA)

# Виводимо перші п'ять рядків
print("Перші п'ять рядків датасету:")
print(df.head())

# Отримуємо базову статистичну інформацію
print("\nСтатистичний опис:")
print(df.describe())

# Перевіряємо наявність пропущених значень
print("\nКількість пропущених значень у кожному стовпці:")
print(df.isnull().sum())

# Перевіряємо типи даних у кожному стовпці
print("\nТипи даних у кожному стовпці:")
print(df.dtypes)

# c) Підготовка даних для аналізу

# Обробка пропущених значень (якщо такі є)
# Оскільки пропущених значень немає, можемо продовжити. Якщо б були, можна було б використати:
# df = df.dropna()

# Кодування категоріальних змінних
# Перетворюємо стовпчик 'Gender' у числові значення: Male=0, Female=1
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Перевіряємо кодування
print("\nДатасет після кодування стовпчика 'Gender':")
print(df.head())

# Застосування методу PCA (Principal Component Analysis):

# a) Застосовуємо PCA до числових ознак датасету

# Вибираємо числові ознаки, видаляючи 'CustomerID' як неінформативний для аналізу
X = df.drop(['CustomerID'], axis=1)

# Масштабуємо дані для коректного застосування PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# b) Визначення оптимальної кількості головних компонент

# Створюємо об'єкт PCA без визначеної кількості компонент
pca = PCA()
pca.fit(X_scaled)

# Отримуємо відсоток поясненої дисперсії для кожної компоненти
explained_variance = pca.explained_variance_ratio_

# Кумулятивна сума поясненої дисперсії
cumulative_variance = explained_variance.cumsum()

# Відображаємо графік кумулятивної поясненої дисперсії
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Кумулятивна пояснена дисперсія')
plt.xlabel('Кількість головних компонент')
plt.ylabel('Кумулятивна пояснена дисперсія')
plt.grid(True)
plt.savefig("practice-6/cumulative_variance.png")
plt.close()

# c) Візуалізація результатів у 2D та 3D просторі

# Вибираємо 2 головні компоненти для 2D візуалізації
pca_2d = PCA(n_components=2)
principal_components_2d = pca_2d.fit_transform(X_scaled)

# Створюємо DataFrame з двома головними компонентами
df_pca_2d = pd.DataFrame(data=principal_components_2d, columns=['PC1', 'PC2'])

# Відображаємо 2D графік
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=df_pca_2d)
plt.title('2D візуалізація PCA')
plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.grid(True)
plt.savefig("practice-6/pca_2d.png")
plt.close()

# Вибираємо 3 головні компоненти для 3D візуалізації
pca_3d = PCA(n_components=3)
principal_components_3d = pca_3d.fit_transform(X_scaled)

# Створюємо DataFrame з трьома головними компонентами
df_pca_3d = pd.DataFrame(data=principal_components_3d, columns=['PC1', 'PC2', 'PC3'])

# Відображаємо 3D графік
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca_3d['PC1'], df_pca_3d['PC2'], df_pca_3d['PC3'])
ax.set_title('3D візуалізація PCA')
ax.set_xlabel('Головна компонента 1')
ax.set_ylabel('Головна компонента 2')
ax.set_zlabel('Головна компонента 3')
plt.savefig("practice-6/pca_3d.png")
plt.close()

# Застосування t-SNE (t-Distributed Stochastic Neighbor Embedding):

# a) Застосовуємо t-SNE до повного набору ознак

# Масштабуємо дані (якщо ще не масштабовані)
# Масштабовані дані зберігаються у змінній X_scaled з попередніх кроків

# Створюємо об'єкт t-SNE з базовими параметрами
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)

# Отримуємо вбудовані представлення даних
X_tsne = tsne.fit_transform(X_scaled)

# Створюємо DataFrame з результатами t-SNE
df_tsne = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])

# b) Експериментуємо з різними параметрами (perplexity, learning_rate)

# Списки параметрів для експериментів
perplexities = [5, 30, 50]
learning_rates = [10, 200, 500]

# Перебираємо різні комбінації параметрів
for perplexity in perplexities:
    for learning_rate in learning_rates:
        # Створюємо об'єкт t-SNE з поточними параметрами
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        df_tsne = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
        
        # c) Візуалізація результатів
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='TSNE1', y='TSNE2', data=df_tsne)
        plt.title(f't-SNE з perplexity={perplexity}, learning_rate={learning_rate}')
        plt.xlabel('TSNE1')
        plt.ylabel('TSNE2')
        plt.grid(True)
        # Зберігаємо графік з унікальною назвою
        filename = f"practice-6/tsne_perp{perplexity}_lr{learning_rate}.png"
        plt.savefig(filename)
        plt.close()

# Порівняння методів PCA та t-SNE

# a) Порівняння результатів

# Створюємо графіки, які показують результати PCA та t-SNE поруч для порівняння
# Візуалізація PCA в 2D
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='PC1', y='PC2', data=df_pca_2d)
plt.title('2D PCA')
plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.grid(True)

# Візуалізація t-SNE з базовими параметрами
plt.subplot(1, 2, 2)
sns.scatterplot(x='TSNE1', y='TSNE2', data=df_tsne)
plt.title('t-SNE (perplexity=30, learning_rate=200)')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.grid(True)

plt.savefig('practice-6/pca_vs_tsne.png')
plt.close()

# b) Аналіз патернів

# Оскільки у нас немає міток класів, можемо застосувати алгоритм кластеризації KMeans
# для виявлення потенційних груп і порівняння їх відображення на PCA та t-SNE

# Визначаємо кількість кластерів (наприклад, 5)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Додаємо мітки кластерів до DataFrame з PCA та t-SNE
df_pca_2d['Cluster'] = labels
df_tsne['Cluster'] = labels

# Візуалізація PCA з кластерами
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='Set1', data=df_pca_2d)
plt.title('2D PCA з кластерами')
plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.legend()
plt.grid(True)

# Візуалізація t-SNE з кластерами
plt.subplot(1, 2, 2)
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', palette='Set1', data=df_tsne)
plt.title('t-SNE з кластерами')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.legend()
plt.grid(True)

plt.savefig('practice-6/pca_vs_tsne_clusters.png')
plt.close()

# Кластеризація на зменшених даних

# a) Застосування алгоритму K-means до даних після PCA та t-SNE
# Встановлюємо кількість кластерів
num_clusters = 5

# Кластеризація на оригінальних масштабованих даних
kmeans_original = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_original.fit(X_scaled)
labels_original = kmeans_original.labels_

# Кластеризація на даних після PCA
kmeans_pca = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_pca.fit(principal_components_2d)
labels_pca = kmeans_pca.labels_

# Кластеризація на даних після t-SNE
kmeans_tsne = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_tsne.fit(X_tsne)
labels_tsne = kmeans_tsne.labels_

# b) Порівняння результатів кластеризації на оригінальних та зменшених даних

# Додаємо мітки кластерів до відповідних DataFrame для візуалізації

# Для оригінальних даних використовуємо PCA для візуалізації
pca_for_viz = PCA(n_components=2)
X_pca_viz = pca_for_viz.fit_transform(X_scaled)
df_original = pd.DataFrame(data=X_pca_viz, columns=['PC1', 'PC2'])
df_original['Cluster'] = labels_original

# Додаємо мітки кластерів до даних після PCA
df_pca_2d['Cluster'] = labels_pca

# Додаємо мітки кластерів до даних після t-SNE
df_tsne['Cluster'] = labels_tsne

# Візуалізація кластерів на оригінальних даних (з використанням PCA для візуалізації)
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='Set1', data=df_original)
plt.title('Кластери на оригінальних даних (PCA візуалізація)')
plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.legend()
plt.grid(True)

# Візуалізація кластерів на даних після PCA
plt.subplot(1, 3, 2)
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='Set1', data=df_pca_2d)
plt.title('Кластери на даних після PCA')
plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.legend()
plt.grid(True)

# Візуалізація кластерів на даних після t-SNE
plt.subplot(1, 3, 3)
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', palette='Set1', data=df_tsne)
plt.title('Кластери на даних після t-SNE')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('practice-6/kmeans_comparison.png')
plt.close()

# Порівняння кластерів за допомогою Adjusted Rand Index
ari_original_pca = adjusted_rand_score(labels_original, labels_pca)
ari_original_tsne = adjusted_rand_score(labels_original, labels_tsne)
ari_pca_tsne = adjusted_rand_score(labels_pca, labels_tsne)

print(f'Adjusted Rand Index між оригінальними даними та PCA: {ari_original_pca:.4f}')
print(f'Adjusted Rand Index між оригінальними даними та t-SNE: {ari_original_tsne:.4f}')
print(f'Adjusted Rand Index між PCA та t-SNE: {ari_pca_tsne:.4f}')