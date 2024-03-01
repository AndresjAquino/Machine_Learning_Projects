# IMPORTACIÓN DE LIBRERIAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# CARGA DEL DATA SET
df = pd.read_csv('/content/registro_temperatura365-ordenado-y-separado.csv')
df.info()

# LIMPIEZA
# Como las variables TMAX y TMIN son tipo object, es posible que haya caracteres no visibles o espacios en blanco en los valores de TMAX y TMIN

# utilizo la función strip() para eliminar espacios en blanco en las cadenas
df = df[df['TMAX'].str.strip() != ""]
df = df[df['TMIN'].str.strip() != ""]
# Convierto a valores numéricos
df['TMAX'] = pd.to_numeric(df['TMAX'])
df['TMIN'] = pd.to_numeric(df['TMIN'])
