# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

#%% importacion de modulos a utilizar

# Una vez invocadas estas funciones, podremos utilizar los módulos a través 
# del identificador que indicamos luego de "as".

# Por ejemplo np.linspace() -> función linspace dentro e NumPy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#%% Declaramos nuestra funcion para la señal 

  
# Datos generales de la simulación, recordar que necesito fs frec de sampling,
#fo frecuencia de la señal y N cantidad total de muestras

fs = 1000.0     # frecuencia de muestreo (Hz)
ts = 1/fs       # tiempo de muestreo
# uno es el reciproco del otro, me dan la misma info en distintos dominios

N = 1000        # cantidad total de muestras

f0 = 1        # frecuencia de la señal

df = fs/N       # resolución espectral, no lo utilizamos en este codigo

# grilla de sampleo temporal 
tt = np.linspace(0, (N-1)*ts, N).flatten()

# linspace( Inicio, Paso, Final) // (N-1)*ts es para definir como es el paso 
# que vamos a tener en la grilla, tenemos 999 puntos distribuidos en nuestro 
# tiempo de sampleo 
    
# Declaro funcion senoidal 
Test_seno = np.sin( 2 * np.pi * f0 * tt  )

plt.figure(1)
line_hdls = plt.plot(tt, Test_seno)
#Para que el gráfico lo genere en una ventana aparte corro la siguiente linea:
# %matplotlib qt
