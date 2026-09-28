"""
Codigo: Bravo_Randolph_Chirino_IDA503_Semana4.py
"""

# Importar librerías necesarias para generar datos, entrenar y evaluar el modelo
from sklearn.datasets import make_classification  # Generar datos de clasificación
from sklearn.model_selection import (
    train_test_split,
)  # Dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.ensemble import (
    RandomForestClassifier,
)  # Importar el modelo Random Forest para clasificación
from sklearn.metrics import accuracy_score  # Evaluar la precisión del modelo
from sklearn.metrics import (
    confusion_matrix,
)  # Evaluar el rendimiento del modelo mediante la matriz de confusión

"""
Generación de datos artificiales
- n_samples=1000: genera 1000 registros.
- n_features=10: cada registro posee 10 características.
- n_classes=2: existen dos clases posibles para clasificar.
- random_state=32: permite obtener resultados reproducibles.
X contiene las características de cada registro e y contiene las etiquetas o clases correspondientes.
"""

# Generar los datos de clasificación
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=32)

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""
Se utilizan 800 registros para entrenar el modelo y 200 registros para evaluar su rendimiento.
"""

# Crear el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

"""
El modelo se construye utilizando 100 árboles de decisión.
La predicción final se obtiene mediante la votación de todos los árboles del bosque.
"""

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

"""
Durante el entrenamiento el modelo aprende patrones presentes en los datos para distinguir las dos clases.
"""

# Realizar predicciones sobre los datos de prueba
y_pred = model.predict(X_test)

"""
El modelo utiliza el conocimiento adquirido durante el entrenamiento para clasificar los registros del conjunto de prueba.
"""

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Mostrar la precisión obtenida
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Generar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión
print("Matriz de confusión:")
print(conf_matrix)

"""Interpretación de resultados
El modelo obtuvo una precisión de 100%, clasificando correctamente todos los registros del conjunto de prueba.
La matriz de confusión muestra que:
- 109 registros de la clase 0 fueron clasificados correctamente.
- 91 registros de la clase 1 fueron clasificados correctamente.
- No se registraron errores de clasificación.
Esto indica que el modelo Random Forest logró identificar correctamente los patrones presentes en los datos generados artificialmente.
"""

# Extras
""" 
Añadimos la matriz de confusión para complementar la evaluación del rendimiento del modelo.
Mientras que la precisión (accuracy) indica el porcentaje total de predicciones correctas, la matriz de confusión permite identificar
en qué clases se producen los aciertos y errores de clasificación.
En este caso, el modelo obtuvo una precisión del 100%, y la matriz de confusión confirma que no existieron errores de clasificación,
ya que todos los registros de las clases 0 y 1 fueron clasificados correctamente.
"""
