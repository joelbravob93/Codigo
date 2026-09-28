"""
Codigo: Bravo_Randolph_Chirino_IDA503_Semana3.py
"""

# Importar librerias
from sklearn.datasets import (
    make_regression,
)  # Generar un conjunto de datos de regresión sintético
from sklearn.model_selection import (
    train_test_split,
)  # Dividir el conjunto de datos en entrenamiento y prueba
from sklearn.tree import (
    DecisionTreeRegressor,
)  # Importar el modelo de árbol de decisión para regresión
from sklearn.metrics import mean_squared_error, r2_score   # Evaluar el rendimiento del modelo
from math import sqrt
import numpy as np
"""
Generación de datos artificiales
-   n_samples: número de muestras a generar
-   n_features: número de características (variables) para cada muestra
-   noise: ruido de los datos
-   random_state: semilla para garantizar la reproducibilidad de los resultados
-   X (información) representa las características de las muestras, mientras que y representa los valores objetivo correspondientes a cada muestra.
-   y (respuesta correcta) es un vector de valores numéricos objetivo, 
mientras que X es una matriz que contiene los valores de las características para cada muestra."""

X, y = make_regression(
    n_samples=1000, n_features=3, noise=0.1, random_state=42
)  # Generar datos artificiales

"""
La función make_regression genera un conjunto de datos artificiales para regresión.
Se crean 1000 registros con 3 características cada uno.
El parámetro noise=0.1 agrega una pequeña cantidad de ruido a los datos.
El parámetro random_state=42 garantiza resultados reproducibles.
"""
# Mostrar dimensiones de los datos generados
print(X.shape)

# Mostrar dimensiones de los valores objetivo
print(y.shape)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""La función train_test_split se utiliza para dividir el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba.
El parámetro test_size=0.2 indica que el 20% (probar rendimiento) de los datos se utilizarán para la prueba, mientras que el 80% (entrenamiento del modelo) 
de los datos  restante se utilizará para el entrenamiento. La semilla random_state se establece en 42 para garantizar que la división de los datos sea reproducible."""

# Mostrar dimensiones del conjunto de entrenamiento
print(
    X_train.shape
)  # 800 muestras para entrenamiento porque es el 80% de 1000 muestras
print(
    y_train.shape
)  # 800 valores objetivo para entrenamiento porque es el 80% de 1000 muestras

# Mostrar dimensiones del conjunto de prueba
print(X_test.shape)  # 200 muestras para prueba porque es el 20% de 1000 muestras
print(y_test.shape)  # 200 valores objetivo para prueba porque es el 20% de 1000 muestras

# Crear el modelo de árbol de decisión
modelo_arbol = DecisionTreeRegressor(random_state=42)

"""Se crea un modelo de árbol de decisión para regresión utilizando DecisionTreeRegressor.
Este modelo será utilizado para aprender patrones numéricos a partir de los datos de entrenamiento.
El parámetro random_state=42 permite que los resultados sean reproducibles."""

# Entrenar el modelo con los datos de entrenamiento
modelo_arbol.fit(X_train, y_train)

"""El método fit se utiliza para entrenar el modelo de árbol de decisión utilizando los datos de entrenamiento (X_train y y_train).
Esto ajusta el modelo a los patrones de los datos, permitiendo que el modelo pueda predecir valores numéricos de las muestras de prueba."""

# Realizar predicciones con el modelo entrenado
y_pred = modelo_arbol.predict(X_test)

"""El método predict se utiliza para realizar predicciones utilizando el modelo entrenado. 
Se pasan los datos de prueba (X_test) al modelo, y este devuelve los valores predichos (y_pred) para cada muestra de prueba."""

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred) # Calcular el error cuadratico medio
rmse = sqrt(mse) # Calcular la raíz del error cuadrático medio para interpretar el error en la misma escala de los datos
r2 = r2_score(y_test, y_pred) # Calcular el coeficiente de determinación R2 del modelo

# Estadísticas del conjunto objetivo
media_y = np.mean(y)
min_y = np.min(y)
max_y = np.max(y)

# Resultados 
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"Media y: {media_y:.4f}")
print(f"Min y: {min_y:.4f}")
print(f"Max y: {max_y:.4f}")

# --------------------------------------------------------
# Interpretación del modelo
# --------------------------------------------------------
"""
El modelo de regresión obtuvo un Error Cuadrático Medio (MSE) de 1042.5360, el cual mide el promedio de los errores al cuadrado entre los valores reales
y los valores predichos.

Para una mejor interpretación, se calcula el Error Cuadrático Medio Raíz (RMSE), obteniendo un valor de 32.2883. Este indicador se encuentra en la misma escala que los datos originales, lo que permite una interpretación más directa del error.

Para contextualizar este valor, se analiza la distribución de los valores objetivo: la media de y es 5.1363, el valor mínimo es -407.7174 y el valor máximo es 485.0544. Considerando este amplio rango de variación, el RMSE representa un error relativamente pequeño en relación con la dispersión total de los datos.

Además, el coeficiente de determinación R² obtenido es 0.9358, lo que indica que el modelo explica aproximadamente el 93.58% de la variabilidad de los datos.

En conjunto, estos resultados permiten concluir que el modelo presenta un buen ajuste y un rendimiento adecuado para la predicción de los valores objetivo.
"""
