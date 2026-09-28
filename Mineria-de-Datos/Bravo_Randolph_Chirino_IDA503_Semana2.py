"""
Codigo: Bravo_Randolph_Chirino_IDA503_Semana2.py
"""

# Importar librerias
from sklearn.datasets import (
    make_classification,
)  # Generar un conjunto de datos de clasificación sintético
from sklearn.model_selection import (
    train_test_split,
)  # Dividir el conjunto de datos en entrenamiento y prueba
from sklearn.tree import (
    DecisionTreeClassifier,
)  # Importar el clasificador de árbol de decisión
from sklearn.metrics import accuracy_score  # Evaluar la precisión del modelo
from sklearn.metrics import (
    confusion_matrix,
)  # Evaluar el rendimiento del modelo utilizando una matriz de confusión

"""
Generación de datos artificiales
-   n_samples: número de muestras a generar
-   n_features: número de características (variables) para cada muestra
-   n_classes: número de clases para la clasificación
-   random_state: semilla para garantizar la reproducibilidad de los resultados
-   X (información) representa las características de las muestras, mientras que y representa las etiquetas de clase correspondientes a cada muestra.
-   y (respuesta correcta) es un vector de etiquetas que indica a qué clase pertenece cada muestra, 
mientras que X es una matriz que contiene los valores de las características para cada muestra."""

X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42
)  # Generar datos artificiales

"""Mediante la función make_classification, se generan 1000 muestras con 20 características cada una, 
pertenecientes a 2 clases diferentes. La semilla random_state se establece en 42 para asegurar que los resultados sean reproducibles."""

# Mostrar dimensiones de los datos generados
print(X.shape)

# Mostrar dimensiones de las etiquetas
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
)  # 800 etiquetas para entrenamiento porque es el 80% de 1000 muestras

# Mostrar dimensiones del conjunto de prueba
print(X_test.shape)  # 200 muestras para prueba porque es el 20% de 1000 muestras
print(y_test.shape)  # 200 etiquetas para prueba porque es el 20% de 1000 muestras

# Crear el modelo de árbol de decisión
modelo_arbol = DecisionTreeClassifier(random_state=42)

"""Se crea un modelo de árbol de decisión de clasificación utilizando DecisionTreeClassifier.
Este modelo será utilizado para aprender patrones a partir de los datos de entrenamiento.
El parámetro random_state=42 permite que los resultados sean reproducibles."""

# Entrenar el modelo con los datos de entrenamiento
modelo_arbol.fit(X_train, y_train)

"""El método fit se utiliza para entrenar el modelo de árbol de decisión utilizando los datos de entrenamiento (X_train y y_train).
Esto ajusta el modelo a los patrones de los datos, permitiendo que el modelo pueda predecir las etiquetas correctas de las muestras de prueba."""

# Realizar predicciones con el modelo entrenado
y_pred = modelo_arbol.predict(X_test)

"""El método predict se utiliza para realizar predicciones utilizando el modelo entrenado. 
Se pasan los datos de prueba (X_test) al modelo, y este devuelve las etiquetas predichas (y_pred) para cada muestra de prueba."""

# Evaluar la precisión del modelo
precision = accuracy_score(y_test, y_pred)
print(
    f"Precisión del modelo en porcentaje: {precision * 100:.2f}%"
)  # Mostrar la precisión en porcentaje con dos decimales

"""La función accuracy_score se utiliza para evaluar la precisión del modelo comparando las etiquetas reales (y_test) con las etiquetas predichas (y_pred).
La precisión se calcula como el número de predicciones correctas dividido por el número total de muestras de prueba.
El resultado se almacena en la variable precision y se muestra en la consola con un formato de dos decimales."""

# Evaluar el rendimiento del modelo utilizando una matriz de confusión
matriz_confusion = confusion_matrix(y_test, y_pred)
print(matriz_confusion)
"""La función confusion_matrix se utiliza para evaluar el rendimiento del modelo utilizando una matriz de confusión.
La matriz de confusión muestra el número de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos, 
lo que proporciona una visión más detallada del rendimiento del modelo en términos de clasificación correcta e incorrecta. 
El resultado se almacena en la variable matriz_confusion y se muestra en la consola."""


""" Explicación de precisión...
El modelo obtuvo una precisión cercana al 88% (87.50%), lo que indica que fue capaz de clasificar correctamente la mayoría de las muestras de prueba.
La matriz de confusión muestra que el modelo presentó pocos errores de clasificación, demostrando un buen rendimiento general.

Clase 0: 86 clasificaciones correctas y 7 errores de clasificación.
Clase 1: 89 clasificaciones correctas y 18 errores de clasificación."""
