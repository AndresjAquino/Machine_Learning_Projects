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

# TRANSFORMO EL DATAFRAME A MATRIZ Y SELECCIONO UNA CIUDAD
matriz = df.values
# Valor a buscar en la columna de nombres
nombre_a_buscar = ' AEROPARQUE AERO'
# Selecciono las filas que contienen el valor deseado en la cuarta columna (índice 3)
matriz_nueva = matriz[matriz[:, 3] == nombre_a_buscar]
matriz_nueva

# AGREGO LAS VARIABLES "Precio helado" Y "Compra helado"
# Defino rango de precios y tasa de incremento
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

# RECONVIERTO A NUEVO DATAFRAME
columnas = ['FECHA', 'TMAX', 'TMIM','NOMBRE','PRECIO_HELADO','COMPRA_HELADO']  # Lista de nombres de columnas
df_nuevo = pd.DataFrame(matriz_nueva, columns=columnas)
df_nuevo.info()
# Convierto a valores numéricos
df_nuevo['FECHA'] = df_nuevo['FECHA'].astype(int)
df_nuevo['TMAX'] = pd.to_numeric(df_nuevo['TMAX'], errors='coerce')
df_nuevo['TMIM'] = pd.to_numeric(df_nuevo['TMIM'], errors='coerce')
df_nuevo['NOMBRE'] = df_nuevo['NOMBRE'].astype(str)
df_nuevo['PRECIO_HELADO'] = pd.to_numeric(df_nuevo['PRECIO_HELADO'], errors='coerce')
df_nuevo['COMPRA_HELADO'] = df_nuevo['COMPRA_HELADO'].astype(str)
df_nuevo.info()
# Elimino la columna 'TMIN'(columna 6) que por algún error que desconozco se duplica
df_nuevo = df_nuevo.drop(df_nuevo.columns[6], axis=1)
df_nuevo.head()
df_nuevo.describe()

# CARGO EL NUEVO DATAFRAME EN UN CSV
# Nombre del archivo y la ubicación a guardar el archivo CSV.
temperaturas_nuevo_csv = 'temperaturas_nuevo.csv'
# Exporto el DataFrame
df_nuevo.to_csv(temperaturas_nuevo_csv, index=False)  # El parámetro 'index=False' evita que se escriban los índices de fila en el archivo CSV


# Ordeno las fechas de forma ascendente
df_ordenado = df_nuevo.sort_values(by='FECHA', ascending=True)
df_ordenado.head(20)
# Genero grafico para observar las temperaturas en función del tiempo
FECHA = df_ordenado['FECHA']
#TMAX = df_ordenado['TMAX']
TMIN = df_ordenado['TMIM']
plt.figure(figsize=(15, 8))
# Trazo las temperaturas máximas
#plt.plot(FECHA, TMAX, label='TMAX', color='red')
# Trazo las temperaturas mínimas
plt.plot(FECHA, TMIN, label='TMIN', color='blue')
plt.xlabel('FECHA')
plt.ylabel('TEMPERATURAS MINIMAS')
plt.legend()
plt.grid(True)
plt.title('FECHA vs TEMPERATURAS MINIMAS')
plt.show()


# R E D   N E U R O N A L

# DIVISIÓN DE LOS DATOS
# Divido el conjunto de datos en conjuntos de entrenamiento y prueba (80% y 20%)
X = df_nuevo['TMIN']
Y = df_nuevo['COMPRA_HELADO']
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42)
# Datos de entrada: Temperatura y Precio del Helado
temperatura = 28.0  # Este valor cambia según la temperatura
precio_helado = 2.5  # Este valor cambia según el precio del helado
# Condiciones
condicion_temperatura = temperatura > 27
condicion_precio = precio_helado < 1.4  # 40 % de incremento

# CREO UN MODELO DE RED NEURONAL UTILIZANDO TENSORFLOW
model = tf.keras.Sequential()
# Agrego tres perceptrones
model.add(tf.keras.layers.Dense(3, activation='sigmoid', input_dim=2))
# Compilo el modelo
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# Evaluo las condiciones y predicciones
entrada = np.array([[condicion_temperatura, condicion_precio]])
prediccion = model.predict(entrada)
# Imprimo la predicción
if prediccion[0][0] >= 0.5:
    print("Es un buen momento para comprar helado.")
else:
    print("No es un buen momento para comprar helado.")
