<img src="https://img.freepik.com/foto-gratis/ai-generated-horses-picture_23-2150650829.jpg?t=st=1697467232~exp=1697467832~hmac=fa391f8f0b9f9b2969c4d727dbf92e82f7c4a7e7472e1fa784c4a21b173aa5fb">

## Tabla de contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Evaluación](#evaluación)
3. [Herramientas Utilizadas](#herramientas-utilizadas)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Cómo usar este proyecto](#cómo-usar-este-proyecto)
6. [Contenido del Jupyter notebook](#contenido-del-jupyter-notebook)
7. [Modelos Utilizados](#modelos-utilizados)


# Predict Health Outcomes of Horses :horse:

### Descripción del Proyecto
Este proyecto utiliza el conjunto de datos disponible en Kaggle (https://www.kaggle.com/competitions/playground-series-s3e22) para realizar un análisis de datos utilizando Python. El objetivo principal es explorar y comprender los datos, así como aplicar técnicas de análisis de datos y aprendizaje automático para predecir el estado de salud de los caballos en estudio en función de diversas características.

### Evaluación :chart_with_upwards_trend:
La métrica que se busca mejorar es el micro-averaged F1-Score. Este valor se calcula usando la totalidad de verdaderos positivos, falsos positivos y falsos negativos, en lugar de calcular el f1 score individualmente para cada clase.

### Herramientas Utilizadas :wrench:
- Python 3.9.17
- Bibliotecas de análisis de datos: Pandas, NumPy.
- Bibliotecas de visualización: Matplotlib, Seaborn.
- Biblioteca de aprendizaje automático: scikit-learn.

### Estructura del Proyecto :open_file_folder:
- train.csv: Archivo CSV que contiene los datos de entrenamiento.
- test.csv: Archivo CSV que contiene los datos de validación.
- horse.csv: Archivo CSV que contiene los datos originales del dataset.
- horses_kaggle.ipynb: Un Jupyter notebook que contiene el código Python para el análisis de datos.
- funciones.py: Archivo Python que contiene las funciones utilizadas para este proyecto.
- submission.csv: Archivo CSV que contiene las predicciones para el archivo test.csv de acuerdo a las instrucciones proporcionadas por Kaggle.

### Cómo usar este proyecto :question:
1. Descarga el conjunto de datos desde Kaggle: https://www.kaggle.com/competitions/playground-series-s3e22/data
2. Coloca los archivos CSV descargados (train.csv, test.csv, original.csv) en la misma carpeta que este proyecto.
3. Abre el Jupyter notebook horses_kaggle.ipynb y ejecuta las celdas de código paso a paso para explorar y analizar los datos.

### Contenido del Jupyter notebook :page_facing_up:
El Jupyter notebook proporciona un análisis completo de los datos, que incluye:
- Exploración de datos: Resumen estadístico, visualización de datos, identificación de valores nulos, etc.
- Preprocesamiento de datos: Limpieza de datos, manejo de valores faltantes, codificación de variables categóricas, etc.
- Análisis de características: Visualización de relaciones entre características.
- Modelado y predicción: Entrenamiento de modelos de aprendizaje automático para predecir estado de salud de los caballos.
- Evaluación del modelo: Evaluación del  micro-averaged F1-Score y rendimiento del modelo.

### Modelos Utilizados :computer:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Support Vector Classifier (SVC)
- Gradient Boosting Classifier
- Bernoulli Naive Bayes
- Linear Discriminant Analysis
- AdaBoost Classifier
- Voting Classifier
