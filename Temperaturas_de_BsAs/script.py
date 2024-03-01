# I M P O R T A C I Ó N   D E   L I B R E R I A S
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# C A R G A   D E L   D A T A   S E T
df = pd.read_csv('/content/registro_temperatura365-ordenado-y-separado.csv')
df
df.info()

# L I M P I E Z A

# Como las variables TMAX y TMIN son tipo object, es posible que haya caracteres no visibles o espacios en blanco en los valores de TMAX y TMIN

# utilizo la función strip() para eliminar espacios en blanco en las cadenas
df = df[df['TMAX'].str.strip() != ""]
df = df[df['TMIN'].str.strip() != ""]
# Convierto a valores numéricos
df['TMAX'] = pd.to_numeric(df['TMAX'])
df['TMIN'] = pd.to_numeric(df['TMIN'])

# T R A N S F O R M A C I O N E S

# Transformo a matriz y selecciono una ciudad
matriz = df.values

# Valor a buscar en la columna de nombres
nombre_a_buscar = ' AEROPARQUE AERO'

# Selecciono las filas que contienen el valor deseado en la cuarta columna (índice 3)
matriz_nueva = matriz[matriz[:, 3] == nombre_a_buscar]
matriz_nueva

# Agrego las variables "Precio helado" y "Compra helado"

# Definir rango de precios y tasa de incremento
rango_precios = (2.0, 5.0)  # Rango de precios de helado
tasa_incremento = 0.1  # Tasa de incremento del precio

# Simulación de regresión lineal en función de la temperatura
temperatura = matriz_nueva[:, 1]  # Supongo que la temperatura está en la segunda columna
precio_base = np.random.uniform(*rango_precios, size=len(matriz_nueva))  # Precio base aleatorio
precio_helado = precio_base + temperatura * tasa_incremento  # Precio del helado

# Agrego la columna "PRECIO DEL HELADO" a la matriz
matriz_nueva = np.column_stack((matriz_nueva, precio_helado))

# Genero la columna "COMPRA HELADO" con probabilidad del 80%
probabilidad_compra = 0.8
precio_base = matriz_nueva[:, -1]  # Supongo que el precio base está en la última columna
compra_helado = np.where(precio_base > 1.4 * precio_helado, "No", "Sí")

# Agrego la columna "COMPRA HELADO" a la matriz
matriz_nueva = np.column_stack((matriz_nueva, compra_helado))

matriz_nueva
