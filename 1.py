import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Частина 1 Підготовка даних
# Завантаження даних
data = pd.read_csv('Mall_Customers.csv')

# Перегляд перших рядків
print(data.head())

# Перевірка на пропущені значення
print(data.isnull().sum())

# Основні статистичні показники
print(data.describe())

# Візуалізація розподілу змінних
data.hist(bins=20, figsize=(10, 8))
plt.tight_layout()
plt.show()

# Вибираємо числові колонки для кластеризації
numerical_data = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Стандартизуємо дані
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)



# Частина 2 Визначення оптимальної кількості кластерів
# Метод ліктя
inertia = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Побудова графіка
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia, marker='o')
plt.xlabel('Кількість кластерів')
plt.ylabel('Інерція')
plt.title('Метод ліктя')
plt.show()


#Коефіцієнт силуету
silhouette_scores = []

for k in cluster_range[1:]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)

# Побудова графіка
plt.figure(figsize=(8, 5))
plt.plot(cluster_range[1:], silhouette_scores, marker='o')
plt.xlabel('Кількість кластерів')
plt.ylabel('Silhouette Score')
plt.title('Аналіз коефіцієнта силуету')
plt.show()



# Частина 3 Кластеризація та аналіз результатів
optimal_k = 5  

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)
# Візуалізація кластерів
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=data['Annual Income (k$)'], 
    y=data['Spending Score (1-100)'], 
    hue=data['Cluster'], 
    palette='tab10', 
    s=100
)
# Додавання центроїдів
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 1], centroids[:, 2], c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.title('Кластери клієнтів')
plt.show()

# Середні значення показників для кожного кластеру
cluster_summary = data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_summary)

# Опис кластерів
for cluster_id, row in cluster_summary.iterrows():
    print(f"Кластер {cluster_id}:")
    print(f"Середній вік: {row['Age']:.1f}")
    print(f"Середній дохід: {row['Annual Income (k$)']:.1f} k$")
    print(f"Середній Spending Score: {row['Spending Score (1-100)']:.1f}")
    print()