# Ingesta
import numpy as np
import pandas as pd
import scipy.stats as stats

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# Modelación
import statsmodels.formula.api as smf
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# Métricas de evaluación
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# Otros
import warnings
warnings.filterwarnings("ignore")

def distribuciones_continuas(data, figsize=(17, 14), filas=4, columnas=3):
    plt.figure(figsize=figsize)
    for index, column in enumerate(data.columns):
        plt.subplot(filas, columnas, index + 1)
        sns.histplot(x=data[column], kde=True, alpha=0.4, palette='YlOrRd')
        plt.axvline(
            data[column].mean(), color="darkorange", linestyle="--", label="Media"
        )
        plt.axvline(data[column].median(), color="red", linestyle="--", label="Mediana")
        plt.axvline(data[column].max(), color="cyan", linestyle="--", label="Máximo")
        plt.axvline(data[column].min(), color="magenta", linestyle="--", label="Mínimo")
        plt.legend(loc="upper right")
        plt.title(f"Histograma '{column}'")
        plt.ylabel("Frecuencia")
    plt.tight_layout()


def distribuciones_categoricas(data, figsize=(17, 16), filas=5, columnas=4):
    plt.figure(figsize=figsize)
    for index, column in enumerate(data.columns):
        plt.subplot(filas, columnas, index + 1)
        sns.histplot(data[column], palette='YlOrRd', alpha=0.6)
        plt.title(f"Gráfico de barras de '{column}'")
        plt.xticks(rotation=30, ha="right")
        plt.xlabel('')
        plt.ylabel('Cantidad')
    plt.tight_layout()


def variable_vs_target(data, var:str, graph_type, figsize=(7,5)):
    if graph_type == sns.countplot:
        plt.figure(figsize=figsize)
        ax = sns.countplot(x=var, hue='outcome', data=data)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 3), textcoords='offset points')
        plt.title(f'{var} vs. outcome')
        plt.xlabel(f'{var}')
        plt.ylabel("Cantidad")
        plt.show()
        
    elif graph_type == sns.boxplot:
        plt.figure(figsize=figsize)
        sns.boxplot(x='outcome', y=var, data=data)
        plt.title(f'Boxplot {var} vs outcome')
        plt.xlabel('Survived')
        plt.ylabel(f'{var}')
        plt.show()

    elif graph_type == sns.histplot:
        plt.figure(figsize=figsize)
        sns.histplot(data=data, x=var, hue='outcome', multiple='stack', alpha=0.7)
        plt.xlabel(f'{var}')
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución de {var}')
        plt.show()


def boxplot_continuos(train_data, data_num, figsize=(25, 22), filas=3, columnas=4):
    plt.figure(figsize=figsize)
    for index, column in enumerate(data_num.columns):
        ax = plt.subplot(filas, columnas, index + 1)
        sns.boxplot(x='outcome', y=column, data=train_data, ax=ax)
        plt.title(f'{column} vs. outcome')
        plt.xticks(rotation=30, ha="right")
        plt.xlabel('')
        plt.ylabel('Cantidad')

        mediana = train_data.groupby('outcome')[column].median()

        for i, tick in enumerate(ax.get_xticks()):
            ax.text(tick, mediana[i], f'{mediana[i]:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()



def countplot_categoricas(train_data, data_cat, figsize=(20, 22), filas=6, columnas=3):
    plt.figure(figsize=figsize)
    for index, column in enumerate(data_cat.columns):
        ax = plt.subplot(filas, columnas, index + 1)
        sns.countplot(x=column, hue='outcome', data=train_data, ax=ax, palette='YlOrRd')
        plt.title(f'{column} vs. outcome')
        plt.xticks(rotation=30, ha="right")
        plt.xlabel('')
        plt.ylabel('Cantidad')
        
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()



def modelacion(modelo, X_train, X_test, y_train, y_test, print_full=False):
    model_tmp_fit = modelo.fit(X_train, y_train)
    yhat_train = model_tmp_fit.predict(X_train.values)
    yhat = model_tmp_fit.predict(X_test.values)

    f1_micro_train = f1_score(y_train, yhat_train, average='micro')
    f1_micro_test = f1_score(y_test, yhat, average='micro')

    if print_full == False:
        print (f'**** Métricas para modelo {modelo} ****\n')
        print('F1-Score Micro-Averaged en train set:', f1_micro_train.round(2),
                '\nF1-Score Micro-Averaged en test set:', f1_micro_test.round(2), '\n','_'*70,'\n')

    else:
        print (f'**** Métricas para modelo {modelo} ****\n')
        print('F1-Score Micro-Averaged en train set:', f1_micro_train.round(2),
                '\nF1-Score Micro-Averaged en test set:', f1_micro_test.round(2))
        

        print(f'\nMétricas en train:\n',
                classification_report(y_train, yhat_train),
                '\n','-'*55,'\n', 
                'Métricas en test:\n',
                classification_report(y_test, yhat),
                '\n','_'*73,'\n')

    return model_tmp_fit, yhat_train, yhat



def plot_importance(fit_model, feat_names, top_n=100):
    tmp_importance = fit_model.feature_importances_
    sort_importance = np.argsort(tmp_importance)[::-1][:top_n]
    names = [feat_names[i] for i in sort_importance]
    plt.title("Feature importance")
    plt.barh(range(len(sort_importance)), tmp_importance[sort_importance])
    plt.yticks(range(len(sort_importance)), names, rotation=0)


def imputar_nulos(data, estrategia='most_frequent'):
    """
    [Resumen]: Recibe un dataframe con np.nan e imputa por la estrategia ingresada
    [data]: dataframe con nulos
    [estrategia]: estrategia de imputacion
    [Returns]:
    [df_imp]: Data con np.nan imputados
    [imputer]: objeto entrenado para imputar np.nan
    """
    # obtener dtypes
    tmp_dict = {}
    for col in data.columns:
        tmp_dict[col] = str(data[col].dtype)

    #imputador
    columnas_nulas = data.columns[data.isnull().any()].tolist()
    imputer = SimpleImputer(strategy=estrategia)
    data_imp = data.copy()
    imputer.fit(data[columnas_nulas])
    data_imp[columnas_nulas] = imputer.transform(data[columnas_nulas])

    #cambiar dtypes
    for col in data.columns:
        data_imp[col] = data_imp[col].astype(tmp_dict[col])
        
    #retorna el df imputado
    return data_imp, imputer