#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 00:10:01 2025

@author: lmaru
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros dados
fs = 1000  # Frecuencia de muestreo en Hz
N = 1000   # Número de muestras
fo = 1    # Frecuencia de la señal en Hz

# Vector de tiempo
t = np.arange(N) / fs  

# Definición de la señal x(t) = sin(2πfo*t)
x = np.sin(2 * np.pi * fo * t)

# Transformada de Fourier (FFT)
X = np.fft.fft(x)  # FFT de la señal
X_magnitud = np.abs(X) / N  # Magnitud normalizada
frecuencias = np.fft.fftfreq(N, d=1/fs)  # Eje de frecuencias

# Solo tomamos la mitad del espectro (parte positiva)
N_half = N // 2
frecuencias = frecuencias[:N_half]
X_magnitud = X_magnitud[:N_half] * 2  # Factor de 2 para compensar la mitad de muestras

# Graficar la señal en el tiempo y el espectro
plt.figure(figsize=(10, 5))

# Señal en el tiempo
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title("Señal en el dominio del tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()
plt.axis([min(t), max(t), -1, 1])#fijo el eje y para que sea igual en cada grafico

# Magnitud del espectro
plt.subplot(2, 1, 2)
plt.plot(frecuencias, X_magnitud)
plt.title("Espectro de amplitud (FFT)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid()
plt.axis([min(frecuencias), max(frecuencias), min(X_magnitud), max(X_magnitud)])#fijo el eje y para que sea igual en cada grafico


plt.tight_layout()
plt.show()
