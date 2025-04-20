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
(t0 , y0) = gen_senoidal(Vmax=1, dc=0, f0=5, ph=0, nn=1000, fs=1000)
#Señal en el lìmite teórico
(t1 , y1) = gen_senoidal(Vmax=1, dc=0, f0=500, ph=np.pi/2, nn=1000, fs=1000)
#Excedidas del límite teórico
(t2 , y2) = gen_senoidal(Vmax=1, dc=0, f0=999, ph=0, nn=1000, fs=1000)
(t3 , y3) = gen_senoidal(Vmax=1, dc=0, f0=1001, ph=0, nn=1000, fs=1000)
(t4 , y4) = gen_senoidal(Vmax=1, dc=0, f0=2001, ph=0, nn=1000, fs=1000)

plt.figure(1)

markerline, stemlines, baseline = plt.stem(t0,y0,basefmt=" ", label='$f_0=5$ hz')
plt.setp(stemlines, linewidth=0.2)  # más fino
plt.setp(markerline, markersize=2,color='black')  # achico los puntos
plt.legend()
plt.grid() # Activa grilla en el gráfico
plt.axis([min(t1), max(t1), -1, 1])#fijo el eje y para que sea igual en cada grafico
plt.title('Señal comodamente muestreada') # Título

plt.figure(2)
plt.subplot(2,1,1)
markerline, stemlines, baseline =  plt.stem(t1, y1, basefmt=" ",label=r'$f_1 = 500\,\mathrm{Hz}$, fase $=\frac{\pi}{2}$')
plt.setp(stemlines, linewidth=0.2)  # más fino
plt.setp(markerline, markersize=2,color='black')  # achico los puntos
plt.legend()
plt.grid() # Activa grilla en el gráfico
plt.axis([min(t1), max(t1), -1, 1])#fijo el eje y para que sea igual en cada grafico
plt.title('Señal muestreada en límite de Niquist') # Título

plt.subplot(2,1,2)
markerline, stemlines, baseline = plt.stem(t2,y2,basefmt=" ", label='$f_2=999$ hz')
plt.setp(stemlines, linewidth=0.5)  # más fino
plt.setp(markerline, markersize=2,color='black')  # achico los puntos
plt.legend()
plt.grid() # Activa grilla en el gráfico
plt.axis([min(t2), max(t2), -1, 1])#fijo el eje y para que sea igual en cada grafico

plt.figure(3)

plt.subplot(2,1,1)
markerline, stemlines, baseline = plt.stem(t3,y3,basefmt=" ", label='$f_3=1001$ hz')
plt.setp(stemlines, linewidth=0.2)  # más fino
plt.setp(markerline, markersize=2,color='black')  # achico los puntos
plt.legend()
plt.grid() # Activa grilla en el gráfico

plt.subplot(2,1,2)
markerline, stemlines, baseline = plt.stem(t4,y4,basefmt=" ", label='$f_4=2001$ hz')
plt.setp(stemlines, linewidth=0.2)  # más fino
plt.setp(markerline, markersize=2,color='black')  # achico los puntos
plt.legend()
plt.grid() # Activa grilla en el gráfico
