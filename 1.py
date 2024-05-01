import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo Excel
data = pd.read_excel("Data_Ev3.xlsx")

# Eliminar filas con valores faltantes
data = data.dropna()

# Dividir los datos en variables independientes (X) y dependientes (Y)
X = data[['YearlyIncome']]  # Variable independiente
y = data['NumberCarsOwned']  # Variable dependiente

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
predictions = model.predict(X_test)

# Calcular el coeficiente de correlación
correlation_coefficient = data.corr().loc['YearlyIncome', 'NumberCarsOwned']

# Crear un DataFrame con los datos de prueba, las predicciones y los errores
df_pred = pd.DataFrame({'Ingreso anual': X_test['YearlyIncome'], 'Número de carros Actual': y_test, 'Número de carros Predicción': predictions})
df_pred['Error'] = df_pred['Número de carros Predicción'] - df_pred['Número de carros Actual']

# Imprimir el cuadro de predicción
print("Cuadro de Predicción:")
print(df_pred)

# Imprimir el coeficiente de correlación
print("\nCoeficiente de Correlación entre Ingreso Anual y Número de Carros:", correlation_coefficient)

# Graficar los resultados
plt.scatter(X_test, y_test, color='blue', label='Datos de prueba')  # Datos de prueba
plt.plot(X_test, predictions, color='red', label='Línea de regresión')  # Línea de regresión
plt.xlabel('Ingreso anual')
plt.ylabel('Número de carros')
plt.title('Regresión lineal: Predicción del número de carros en función del ingreso anual')
plt.legend()
plt.show()
