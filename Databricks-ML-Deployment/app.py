from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(
    __name__
)  # Se crea una instancia de la clase Flask, que representa la aplicación web

modelo = joblib.load(
    "modelo_entrenado.pkl"
)  # Se carga el modelo previamente entrenado desde un archivo .pkl utilizando joblib


@app.route(
    "/", methods=["GET", "POST"]
)  # Se define la ruta principal de la aplicación y se especifica que acepta métodos GET y POST
def inicio():  # Se define la función inicio que se ejecutará cuando se acceda a la ruta principal

    prediccion = None

    if (
        request.method == "POST"
    ):  # Se verifica si el método de la solicitud es POST, lo que indica que se enviaron datos desde un formulario

        mean_radius = float(
            request.form["mean_radius"]
        )  # Se obtienen los valores del formulario y se convierten a float
        mean_texture = float(
            request.form["mean_texture"]
        )  # Se obtienen los valores del formulario y se convierten a float
        mean_area = float(
            request.form["mean_area"]
        )  #  Se obtienen los valores del formulario y se convierten a float
        mean_concavity = float(
            request.form["mean_concavity"]
        )  # Se obtienen los valores del formulario y se convierten a float

        datos = pd.DataFrame(
            [
                {
                    "mean_radius": mean_radius,  # Se crea un diccionario con los valores obtenidos
                    "mean_texture": mean_texture,  # Se crea un diccionario con los valores obtenidos
                    "mean_area": mean_area,  # Se crea un diccionario con los valores obtenidos
                    "mean_concavity": mean_concavity,  # Se crea un diccionario con los valores obtenidos
                }
            ]
        )

        resultado = modelo.predict(datos)[
            0
        ]  # Se realiza la predicción utilizando el modelo cargado y se obtiene el resultado

        if resultado == 1:  # Se verifica el resultado de la predicción
            prediccion = (
                "Benigno"  # Se asigna el valor "Benigno" a la variable prediccion
            )
        else:  # Se verifica el resultado de la predicción
            prediccion = (
                "Maligno"  # Se asigna el valor "Maligno" a la variable prediccion
            )

    return render_template(
        "index.html", prediccion=prediccion
    )  # Se renderiza la plantilla index.html y se pasa la variable prediccion como contexto


if __name__ == "__main__":  # Se verifica si el archivo se está ejecutando directamente
    app.run(debug=True)  # Se ejecuta la aplicación en modo de depuración
