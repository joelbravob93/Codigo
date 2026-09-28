# Inteligencia Artificial y Machine Learning – Semana 4 🤖

Proyecto académico desarrollado en el módulo **Inteligencia Artificial y Machine Learning**, enfocado en la utilización de **Databricks** para el procesamiento de datos y la generación de un modelo predictivo, junto con su posterior integración en una aplicación web.

## 🎯 Objetivo

Desarrollar un flujo básico de implementación de un modelo de Machine Learning, considerando el procesamiento de datos, entrenamiento y almacenamiento del modelo, además de su integración con una aplicación backend y frontend.

## 🧩 Descripción del proyecto

El proyecto se desarrolló inicialmente en **Databricks**, donde se realizó el proceso de preparación y transformación de los datos (**ETL**) y posteriormente el entrenamiento del modelo predictivo.

Una vez finalizado el entrenamiento, el modelo fue descargado y almacenado localmente como:

`modelo_entrenado.pkl`

Posteriormente, se desarrolló una aplicación web utilizando **Flask**, encargada de cargar el modelo previamente entrenado y utilizarlo para generar predicciones a partir de los datos ingresados por el usuario.

## 🔄 Flujo del proyecto

```text
Datos
  ↓
ETL en Databricks
  ↓
Preparación de datos
  ↓
Entrenamiento del modelo
  ↓
Evaluación
  ↓
modelo_entrenado.pkl
  ↓
Backend Flask
  ↓
Frontend
  ↓
Ingreso de datos
  ↓
Predicción