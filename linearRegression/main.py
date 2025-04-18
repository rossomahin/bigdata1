import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1) Загрузка датасета Ириса Фишера
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 2) Построение линейных регрессий для каждого целевого значения и признака
best_r2 = 0
best_i, best_j = 0, 0
best_reg = None

for j in range(4):
    for i in range(4):
        if i != j:
            reg = LinearRegression()
            reg.fit(X[:, i].reshape(-1, 1), X[:, j])
            r2 = reg.score(X[:, i].reshape(-1, 1), X[:, j])
            if r2 > best_r2:
                best_r2 = r2
                best_i, best_j = i, j
                best_reg = reg

# 3) Выбор наилучшей регрессии
print(f"Наилучшая регрессия: признак {feature_names[best_i]} -> признак {feature_names[best_j]}")
print(f"Коэффициент детерминации R^2: {best_r2:.2f}")

# 4) Вывод графика для лучшей регрессии
plt.figure(figsize=(8, 6))
plt.scatter(X[:, best_i], X[:, best_j], c=y, cmap='viridis')
plt.plot(X[:, best_i], best_reg.predict(X[:, best_i].reshape(-1, 1)), color='r')
plt.xlabel(feature_names[best_i])
plt.ylabel(feature_names[best_j])
plt.title(f"Линейная регрессия: {feature_names[best_i]} -> {feature_names[best_j]}")
plt.show()

# 5) Построение регрессий с несколькими признаками
best_r2 = 0
best_i1, best_i2, best_j = 0, 0, 0
best_reg = None

for j in range(4):
    for i1 in range(4):
        for i2 in range(i1 + 1, 4):
            if i1 != j and i2 != j:
                reg = LinearRegression()
                reg.fit(X[:, [i1, i2]], X[:, j])
                r2 = reg.score(X[:, [i1, i2]], X[:, j])
                if r2 > best_r2:
                    best_r2 = r2
                    best_i1, best_i2, best_j = i1, i2, j
                    best_reg = reg

print(f"Наилучшая регрессия с несколькими признаками: {feature_names[best_i1]}, {feature_names[best_i2]} -> {feature_names[best_j]}")
print(f"Коэффициент детерминации R^2: {best_r2:.2f}")

