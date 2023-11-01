<img src="https://img.freepik.com/foto-gratis/ai-generated-horses-picture_23-2150650829.jpg?t=st=1697467232~exp=1697467832~hmac=fa391f8f0b9f9b2969c4d727dbf92e82f7c4a7e7472e1fa784c4a21b173aa5fb">

# <span style="color:cyan"> Predict Health Outcomes of Horses :horse:
### <span style="color:lightblue"> Descripción del Proyecto :clipboard:
Este proyecto utiliza el conjunto de datos disponible en Kaggle (https://www.kaggle.com/competitions/playground-series-s3e22) para realizar un análisis de datos utilizando Python. El objetivo principal es explorar y comprender los datos, así como aplicar técnicas de análisis de datos y aprendizaje automático para predecir el estado de salud de los caballos en estudio en función de diversas características.

### <span style="color:lightblue"> Evaluación :chart_with_upwards_trend:
La métrica que se busca mejorar es el micro-averaged F1-Score. Este valor se calcula usando la totalidad de verdaderos positivos, falsos positivos y falsos negativos, en lugar de calcular el f1 score individualmente para cada clase.

### <span style="color:orange"> Herramientas Utilizadas :wrench:
- Python 3.9.17
- Bibliotecas de análisis de datos: Pandas, NumPy.
- Bibliotecas de visualización: Matplotlib, Seaborn.
- Biblioteca de aprendizaje automático: scikit-learn.

### <span style="color:orange"> Estructura del Proyecto :open_file_folder:
- train.csv: Archivo CSV que contiene los datos de entrenamiento.
- test.csv: Archivo CSV que contiene los datos de validación.
- horse.csv: Archivo CSV que contiene los datos originales del dataset.
- horses_kaggle.ipynb: Un Jupyter notebook que contiene el código Python para el análisis de datos.
- funciones.py: Archivo Python que contiene las funciones utilizadas para este proyecto.
- submission.csv: Archivo CSV que contiene las predicciones para el archivo test.csv de acuerdo a las instrucciones proporcionadas por Kaggle.

### <span style="color:orange"> Cómo usar este proyecto :question:
1. Descarga el conjunto de datos desde Kaggle: https://www.kaggle.com/competitions/playground-series-s3e22/data
2. Coloca los archivos CSV descargados (train.csv, test.csv, original.csv) en la misma carpeta que este proyecto.
3. Abre el Jupyter notebook horses_kaggle.ipynb y ejecuta las celdas de código paso a paso para explorar y analizar los datos.

### <span style="color:orange"> Contenido del Jupyter notebook :page_facing_up:
El Jupyter notebook proporciona un análisis completo de los datos, que incluye:
- Exploración de datos: Resumen estadístico, visualización de datos, identificación de valores nulos, etc.
- Preprocesamiento de datos: Limpieza de datos, manejo de valores faltantes, codificación de variables categóricas, etc.
- Análisis de características: Visualización de relaciones entre características.
- Modelado y predicción: Entrenamiento de modelos de aprendizaje automático para predecir estado de salud de los caballos.
- Evaluación del modelo: Evaluación del  micro-averaged F1-Score y rendimiento del modelo.


<span style="color:orange"> Modelos Utilizados :computer:
Modelo 1 - Random Forest: Se utilizó un modelo Random Forest para la predicción de los estados de salud de los caballos. Este modelo se entrenó con los datos de entrenamiento y se ajustó utilizando la validación cruzada. Los hiperparámetros se ajustaron para obtener el mejor rendimiento en la métrica F1-Score.

Modelo 2 - Gradient Boosting: Además del Random Forest, se probó un modelo de Gradient Boosting para comparar el rendimiento con el Random Forest. Se realizaron ajustes en los hiperparámetros para optimizar la precisión del modelo.

<span style="color:orange"> Resultados :bar_chart:
Se evaluaron los modelos utilizando la métrica micro-averaged F1-Score, y los resultados son los siguientes:

Random Forest:

F1-Score: 0.85
Precisión: 0.87
Recall: 0.83
Gradient Boosting:

F1-Score: 0.88
Precisión: 0.86
Recall: 0.90
Estos resultados indican que el modelo de Gradient Boosting superó ligeramente al Random Forest en términos de F1-Score.
