import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter

# Загружаем данные о цветках ириса
iris = load_iris()
X = iris.data  # Признаки (длина и ширина чашелистика и лепестка)
y = iris.target  # Метки (сорта ирисов)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Диапазон количества кластеров для оценки
k_range = range(1, 21)
accuracy_scores = []  # Список для хранения точности для каждого k
best_accuracy = 0  # Лучшая точность
best_k = 0  # Оптимальное количество кластеров

# Цикл по возможным значениям k
for k in k_range:
    # Создаем объект KMeans с заданным числом кластеров
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)  # Обучаем модель на обучающих данных
    y_pred = kmeans.predict(X_test)  # Предсказываем метки для тестовых данных

    # Инициализируем массив меток для тестовых данных
    labels = np.zeros_like(y_pred)
    
    # Присваиваем метки на основе наиболее частых меток среди данных в кластере
    for i in range(k):
        mask = (y_pred == i)  # Маска для элементов, принадлежащих кластеру i
        if np.any(mask):
            # Находим наиболее частую метку среди элементов в кластере
            most_common = Counter(y_test[mask]).most_common(1)
            if most_common:
                # Присваиваем метке соответствующую метку из множества реальных данных
                labels[mask] = most_common[0][0]

    # Вычисляем точность предсказания
    accuracy = accuracy_score(y_test, labels)
    accuracy_scores.append(accuracy)

    # Обновляем наилучшую точность и соответствующее значение k
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# Вычисляем среднюю ошибку на тестовой выборке
average_error = 1 - best_accuracy

# Выводим оптимальное количество кластеров и ошибку
print(f"Оптимальное значение k: {best_k}")
print(f"Средняя ошибка на тестовой выборке: {average_error:.2f}")

# Визуализация результатов

# Настроим размер графиков
plt.figure(figsize=(12, 6))

# Левый график: Реальные сорта ирисов (цвета тестовых данных)
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
plt.title("Реальные сорта ирисов")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# Правый график: Результаты кластеризации с оптимальным k
plt.subplot(1, 2, 2)
kmeans = KMeans(n_clusters=best_k, random_state=42)  # Инициализируем KMeans с оптимальным k
kmeans.fit(X_train)  # Обучаем модель на обучающих данных
y_pred = kmeans.predict(X_test)  # Предсказываем метки для тестовых данных

# Инициализируем массив меток для тестовых данных
labels = np.zeros_like(y_pred)

# Присваиваем метки на основе наиболее частых меток среди данных в кластере
for i in range(best_k):
    mask = (y_pred == i)  # Маска для элементов, принадлежащих кластеру i
    if np.any(mask):
        # Находим наиболее частую метку среди элементов в кластере
        most_common = Counter(y_test[mask]).most_common(1)
        if most_common:
            # Присваиваем метке соответствующую метку из множества реальных данных
            labels[mask] = most_common[0][0]

# Визуализируем результаты кластеризации
plt.scatter(X_test[:, 0], X_test[:, 1], c=labels, cmap='viridis')
plt.title("Результаты кластеризации KMeans")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# Показываем графики
plt.tight_layout()
plt.show()
