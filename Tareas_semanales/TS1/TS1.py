# -*- coding: utf-8 -*-
"""
Created on Spyder

@author: Lucas Marú

Descripción:
------------
Generador de senoidales.
En este primer trabajo comenzaremos por diseñar un generador de señales que 
utilizaremos en las primeras simulaciones que hagamos. La primer tarea 
consistirá en programar una función que genere señales senoidales y que permita 
parametrizar:

la amplitud máxima de la senoidal (volts)
su valor medio (volts)
la frecuencia (Hz)
la fase (radianes)
la cantidad de muestras digitalizada por el ADC (# muestras)
la frecuencia de muestreo del ADC.

Es decir que la función que uds armen debería admitir ser llamada de la 
siguiente manera:

tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs)
Recuerden que tanto xx como tt deben ser vectores de Nx1. Puede resultarte útil 
el módulo de visualización matplotlib.pyplot donde encontrarán todas las 
funciones de visualización estilo Matlab. Para usarlo:

import matplotlib.pyplot as plt
plt.plot(tt, xx)

"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Declaramos nuestra funcion
def gen_senoidal(Vmax, dc, f0, ph, nn, fs):
    #el tiempo de sampleo me lo da la fs
    ts=1/fs
    #armo el eje temporal, no termino de entende "(N-1)*ts"
    tt = np.linspace(0, (nn-1)*ts, nn).flatten()
    sen_gen = Vmax * np.sin(2 * np.pi *f0 * tt + ph)
    return (tt , sen_gen)

#señal cómodamente muestreada según el criterio que definimos
#(t0 , y0) = gen_senoidal(Vmax=1, dc=0, f0=100, ph=0, nn=1000, fs=1000)
#Señal en el lìmite teórico
(t1 , y1) = gen_senoidal(Vmax=1, dc=0, f0=500, ph=0, nn=1000, fs=1000)
#Excedidas del límite teórico
(t2 , y2) = gen_senoidal(Vmax=1, dc=0, f0=999, ph=0, nn=1000, fs=1000)
(t3 , y3) = gen_senoidal(Vmax=1, dc=0, f0=1001, ph=0, nn=1000, fs=1000)
(t4 , y4) = gen_senoidal(Vmax=1, dc=0, f0=2001, ph=0, nn=1000, fs=1000)

"""plt.figure(1)
plt.subplot(2,3,1)
seno0 = plt.plot(t0, y0)
plt.grid() # Activa grilla en el gráfico
plt.title('fo=100hz') # Título"""

plt.subplot(2,2,1)
seno1 = plt.plot(t1, y1)
plt.grid() # Activa grilla en el gráfico
plt.title('fo=500hz') # Título
plt.axis([min(t1), max(t1), -1, 1])#fijo el eje y para que sea igual en cada grafico

plt.subplot(2,2,2)
seno2 = plt.plot(t2, y2)
plt.grid() # Activa grilla en el gráfico
plt.title('fo=999hz') # Título

plt.subplot(2,2,3)
seno3 = plt.plot(t3, y3)
plt.grid() # Activa grilla en el gráfico
plt.title('fo=1001hz') # Título

plt.subplot(2,2,4)
seno4 = plt.plot(t4, y4)
plt.grid() # Activa grilla en el gráfico
plt.title('fo=2001hz') # Título
