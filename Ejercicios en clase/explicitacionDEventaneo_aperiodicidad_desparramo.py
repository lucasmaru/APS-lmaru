#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización de una senoidal con su ventana rectangular marcada en los bordes
@author: lmaru
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def genera_grafica_tiempo_espectro(f0,fs,N):
    #%% Datos de la simulación
    ts = 1/fs # tiempo de muestreo
    df = fs/N # resolución espectral
    tt = np.linspace(0, (N-1)*ts, N).flatten()
    "quiero que tt arranque en cero, termine en 0,999 y tenga N elementos"
    A = math.sqrt(2) # proceso de normalización 
    xx = A*np.sin( 2 * np.pi * f0 * tt ) # Declaro funcion senoidal
    plt.figure()
    plt.subplot(2,1,1)
    markerline, stemlines, baseline = plt.stem(tt, xx,basefmt=" ")
    plt.setp(stemlines, linewidth=0.5)  # más fino
    plt.setp(markerline, markersize=2,color='black')  # achico los puntos
    plt.title(f"Señal de {f0} Hz muestreada a {fs} Hz (tiempo)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    #plt.axis([min(tt), max(tt), -1, 1])#fijo el eje y para que sea igual en cada grafico
    ft= 1/N*np.fft.fft(xx)#escalamiento no es importante (1/N)
    ff = np.linspace(0, (N-1)*df, N)# grilla de sampleo frecuencial
    bfrec = ff <= fs/2#vector booleano mitad True, mitad False
    plt.subplot(2,1,2)
    #plt.plot( ff[bfrec], np.abs(ft[bfrec]))
    # plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft[bfrec])**2))
    markerline, stemlines, baseline = plt.stem(ff[bfrec], 10 * np.log10(2 * np.abs(ft[bfrec])**2), basefmt=" ")
    #captura los objetos que devuelve stem para por ajustarlos con setp
    plt.setp(stemlines, linewidth=0.5)  # más fino
    plt.setp(markerline, markersize=4,color='black')  # achico los puntos
    # Línea fina que une los puntos (por encima del stem)
    plt.plot( ff[bfrec], 10 * np.log10(2 * np.abs(ft[bfrec])**2), color='red', linewidth=2, label='Interpolación visual')    
    plt.title("Espectro (frecuencia)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (dB)")
    plt.grid(True)
    return tt,ts,N,f0,fs

tt ,ts , N ,f0 ,fs= genera_grafica_tiempo_espectro(f0=4, fs=1000 , N=1000)

window = np.ones(N)
plt.subplot(2,1,1)
plt.plot(tt, window, label='Ventana rectangular', color='black', linestyle='--')
# Líneas verticales en los extremos del cajón
plt.vlines(tt[0], ymin=0, ymax=1, color='black', linestyle='--')
plt.vlines(tt[-1], ymin=0, ymax=1, color='black', linestyle='--')
plt.title(f"Senoidal de {f0} Hz muestreada a {fs} Hz (tiempo)y ventana rectangular")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

tt ,ts , N ,f0 ,fs = genera_grafica_tiempo_espectro(f0=4, fs=1200 , N=4000)

window = np.ones(N)
plt.subplot(2,1,1)
plt.plot(tt, window, label='Ventana rectangular', color='black', linestyle='--')
# Líneas verticales en los extremos del cajón
plt.vlines(tt[0], ymin=0, ymax=1, color='black', linestyle='--')
plt.vlines(tt[-1], ymin=0, ymax=1, color='black', linestyle='--')
plt.title(f"Senoidal de {f0} Hz muestreada a {fs} Hz (tiempo)y ventana rectangular")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
