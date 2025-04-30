#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 19:14:02 2025

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
    plt.plot(tt, xx)
    # markerline, stemlines, baseline = plt.stem(
    #     tt, xx, basefmt=" ", linefmt='C0-', markerfmt='C1o', use_line_collection=True
    # )
    # plt.setp(stemlines, linewidth=0.7)  # más fino
    # plt.setp(markerline, markersize=6, markerfacecolor='black', markeredgewidth=1)
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
    plt.plot( ff[bfrec], 20* np.log10(2*np.abs(ft[bfrec])))
    plt.title("Espectro (frecuencia)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (dB)")
    plt.grid(True)

#genera_grafica_tiempo_espectro(1, 1000,1000)
#genera_grafica_tiempo_espectro(1, 1000.5,1000)
genera_grafica_tiempo_espectro(f0=0.5, fs=1000 , N=1000)
#genera_grafica_tiempo_espectro(f0=4, fs=1200 , N=400)
#genera_grafica_tiempo_espectro(1, 51.5, 100)
