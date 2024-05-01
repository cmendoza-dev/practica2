import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar los datos desde el archivo Excel
data = pd.read_excel("Data_Ev3.xlsx")

# Eliminar filas con valores faltantes
data = data.dropna()

# Dividir los datos en variables independientes (X) y dependiente (Z)
X = data[['YearlyIncome', 'TotalChildren']]  # Variables independientes
y = data['NumberCarsOwned']  # Variable dependiente

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
predictions = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = model.score(X_test, y_test)
print("Precisión del modelo:", accuracy)

# Obtener los coeficientes del modelo
coeficientes = model.coef_
intercepto = model.intercept_

# Imprimir los coeficientes
print("Coeficientes del modelo:")
print("Intercepto:", intercepto)
print("Coeficiente para YearlyIncome:", coeficientes[0])
print("Coeficiente para TotalChildren:", coeficientes[1])

# Crear un DataFrame para comparar los valores reales y las predicciones
comparacion = pd.DataFrame({'Real': y_test, 'Predicción': predictions})
print("\nTabla de comparación entre valores reales y predicciones:")
print(comparacion.head(10))  # Mostrar las primeras 10 filas

# Graficar los resultados
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# Graficar los datos de prueba
ax.scatter(X_test['YearlyIncome'], X_test['TotalChildren'], y_test, color='blue', label='Datos de prueba')

# Graficar el plano de predicción
x_surf, y_surf = np.meshgrid(np.linspace(X_test['YearlyIncome'].min(), X_test['YearlyIncome'].max(), 100),
                             np.linspace(X_test['TotalChildren'].min(), X_test['TotalChildren'].max(), 100))
exog = pd.DataFrame({'YearlyIncome': x_surf.ravel(), 'TotalChildren': y_surf.ravel()})
out = model.predict(exog)
ax.plot_surface(x_surf, y_surf, out.reshape(x_surf.shape), color='red', alpha=0.5)

ax.set_xlabel('Ingreso anual')
ax.set_ylabel('TotalChildren')
ax.set_zlabel('Número de carros')

plt.title('Regresión lineal múltiple: Predicción del número de carros en función del Ingreso anual y Total de hijos')
plt.legend()
plt.show()
