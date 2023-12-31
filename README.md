# Predict Health Outcomes of Horses :horse:

<img src="https://img.freepik.com/foto-gratis/ai-generated-horses-picture_23-2150650829.jpg?t=st=1697467232~exp=1697467832~hmac=fa391f8f0b9f9b2969c4d727dbf92e82f7c4a7e7472e1fa784c4a21b173aa5fb">

## Tabla de contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto-clipboard)
2. [Evaluación](#evaluación-chart_with_upwards_trend)
3. [Herramientas Utilizadas](#herramientas-utilizadas-wrench)
4. [Estructura del Proyecto](#estructura-del-proyecto-open_file_folder)
5. [Cómo usar este proyecto](#cómo-usar-este-proyecto-question)
6. [Contenido del Jupyter notebook](#contenido-del-jupyter-notebook-page_facing_up)
7. [Modelos Utilizados](#modelos-utilizados-computer)
8. [Resultados](#resultados-bar_chart)


### Descripción del Proyecto :clipboard:
Este proyecto utiliza el conjunto de datos disponible en Kaggle (https://www.kaggle.com/competitions/playground-series-s3e22) para realizar un análisis de datos utilizando Python. El objetivo principal es explorar y comprender los datos, así como aplicar técnicas de análisis de datos y aprendizaje automático para predecir el estado de salud de los caballos en estudio en función de diversas características.

### Evaluación :chart_with_upwards_trend:
La métrica que se busca mejorar es el micro-averaged F1-Score. Este valor se calcula usando la totalidad de verdaderos positivos, falsos positivos y falsos negativos, en lugar de calcular el f1 score individualmente para cada clase.

### Herramientas Utilizadas :wrench:
- Python 3.9.17
- Bibliotecas de análisis de datos: Pandas, NumPy.
- Bibliotecas de visualización: Matplotlib, Seaborn.
- Biblioteca de aprendizaje automático: scikit-learn.

### Estructura del Proyecto :open_file_folder:
- **train.csv:** Archivo CSV que contiene los datos de entrenamiento.
- **test.csv:** Archivo CSV que contiene los datos de validación.
- **horse.csv:** Archivo CSV que contiene los datos originales del dataset.
- **horses_kaggle.ipynb:** Un Jupyter notebook que contiene el código Python para el análisis de datos.
- **funciones.py:** Archivo Python que contiene las funciones utilizadas para este proyecto.
- **submission.csv:** Archivo CSV que contiene las predicciones para el archivo `test.csv` de acuerdo a las instrucciones proporcionadas por Kaggle.

### Cómo usar este proyecto :question:
1. Asegúrate de tener instalado Python 3.9.17 en tu sistema.
2. Descarga el conjunto de datos desde Kaggle: https://www.kaggle.com/competitions/playground-series-s3e22/data
3. Coloca los archivos CSV descargados (`train.csv`, `test.csv`, `original.csv`) en la misma carpeta que este proyecto.
4. Abre el Jupyter notebook `horses_kaggle.ipynb` y ejecuta las celdas de código paso a paso para explorar y analizar los datos.

### Contenido del Jupyter notebook :page_facing_up:
El Jupyter notebook proporciona un análisis completo de los datos, que incluye:
- Exploración de datos: Resumen estadístico, visualización de datos, identificación de valores nulos, etc.
- Preprocesamiento de datos: Limpieza de datos, manejo de valores faltantes, codificación de variables categóricas, etc.
- Análisis de características: Visualización de relaciones entre características.
- Modelado y predicción: Entrenamiento de modelos de aprendizaje automático para predecir estado de salud de los caballos.
- Evaluación del modelo: Evaluación del  micro-averaged F1-Score y rendimiento del modelo.

### Modelos Utilizados :computer:
- Logistic Regression
- K-Nearest Neighbors Classifier
- Random Forest Classifier
- Support Vector Classifier
- Gradient Boosting Classifier
- Bernoulli Naive Bayes
- Linear Discriminant Analysis
- AdaBoost Classifier
- Voting Classifier

### Resultados :bar_chart:
Se evaluaron todos los modelos utilizando la métrica micro-averaged F1-Score, y los resultados son los siguientes:

- Logistic Regression: F1-Score: 0.7
- K-Nearest Neighbors Classifier: F1-Score: 0.68
- Random Forest Classifier: F1-Score: 0.73
- Support Vector Classifier: F1-Score: 0.71
- Gradient Boosting Classifier: F1-Score: 0.74
- Bernoulli NB: F1-Score: 0.68
- Linear Discriminant Analysis: F1-Score: 0.7
- AdaBoost Classifier: F1-Score: 0.7
- Voting Classifier: F1-Score: 0.75

Para el Voting Classifier se hizo una combinación de los dos mejores modelos, logrando reducir el overfitting y así obtener un mejor desempeño del modelo sobre nuevos datos.
