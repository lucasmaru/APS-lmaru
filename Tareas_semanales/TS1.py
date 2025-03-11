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

#%% Declaramos nuestra funcion
def gen_senoidal(Vmax, dc, ff, ph, nn, fs):
    #el tiempo de sampleo me lo da la fs
    ts=1/fs
    #armo el eje temporal, no termino de entende "(N-1)*ts"
    tt = np.linspace(0, (nn-1)*ts, nn).flatten()
    sen_gen = Vmax * np.sin(2 * np.pi *ff * tt + ph)
    return (tt , sen_gen)

(t , y) = gen_senoidal(Vmax=1, dc=0, ff=100, ph=0, nn=1000, fs=1000)
plt.figure(1)
seno = plt.plot(t, y)
