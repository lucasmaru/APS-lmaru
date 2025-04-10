#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 18:44:34 2025

@author: lmaru"""

#%%
"""
TAREA SEMANAL 4 - Primeras nociones de la estimacion espectral
La estimación espectral es una técnica utilizada en el procesamiento de señales para determinar cómo se
distribuye la potencia de una señal en función de la frecuencia
"""

#%% módulos y funciones a importar
import numpy as np
from scipy.signal.windows import hamming, hann, blackman, kaiser
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

#%% Genero mi Omega 1 que va a ser una frecuencia original mas una inceretidumbre en la frecuencia
# Esto genera que no sepa la frecuencia que estamos, sino que solo sepamos entre que valores va a estar

#%% Datos de la simulacion

fs = 1000           # frecuencia de muestreo (Hz)
N = 1000            # cantidad de muestras
ts = 1/fs           # tiempo de muestreo
df = fs/N           # resolución espectral

N_Test = 200        # Numero de pruebas

SNR = 10            # Signal to Noise Ratio 

Sigma2 = 10**-1     # Despejando el PN potencia de ruido de la ecuacion de SNR llego a este valor

Omega_0 = fs/4      # Nos ponemos a mitad de banda digital

#%% Genero mi vector de 1000x200 de la senoidal

A1 = np.sqrt(2)                                             # Genero la amplitud de manera que el seno quede normalizado

fr = np.random.uniform(-1/2,1/2,N_Test).reshape(1,N_Test)      # Declaro mi vector 1x200 al pasar 200 valores a la uniforme     
                                                            # Fuerzo las dimensiones
Omega_1 = Omega_0 + fr*df                                   # Genero mi Omega 1

# Genero vector de tiempo para poder meterlo como mi vector de 1000 x 200 en el seno 
tt = np.linspace(0, (N-1)*ts, N).reshape(N,1)

tt = np.tile(tt, (1, N_Test))                               # Genero la matriz de 1000 x 200

# Al mutiplicar con * hacemos el producto matricial para que quede de 1000x200
S = A1 * np.sin(2 * np.pi * Omega_1 * tt)

#%% Genereo el ruido para la señal

# Para poder general la señal de ruido, tenemos que tener una distribucion normal con un N(o,sigma)

Media = 0                   # Media
SD_Sigma = np.sqrt(Sigma2)  # Desvio standar 

nn = np.random.normal(Media, SD_Sigma, N).reshape(N,1)              # Genero señal de ruido

nn = np.tile(nn, (1,N_Test))                                        # Ahora tengo que generaer mi matriz de ruido de 200x1000


#%% Ahora genero mi señal final sumando las matrices

# Esto seria mi x(k) = a1 * sen(W1 * k ) + n(k), siendo N(k) el ruido
Signal = S + nn

#%% Calcular la FFT de cada señal en la matriz Signal

X_f = fft(Signal, axis=0)  # FFT en cada columna (cada señal)
X_f = fftshift(X_f, axes=0)  # Centramos el espectro
X_f_norm = X_f/np.max(np.abs(X_f))
# Generamos el eje de frecuencias
frec = np.linspace(-fs/2, fs/2, N)  # Eje de frecuencias

#%% Graficamos la magnitud de la FFT para algunas señales

num_senales = 10  # Número de señales a graficar
indices = np.linspace(0, N_Test-1, num_senales, dtype=int)  # Seleccionamos señales espaciadas

plt.figure(figsize=(10, 6))

for i in indices:
    #plt.plot(frec, np.abs(X_f[:, i]), label=f'Señal {i+1}')  # Magnitud de la FFT
    plt.plot(frec, 10 * np.log10(2 * np.abs(X_f_norm[:, i])**2), label=f'Señal {i+1}')  # Magnitud de la FFT

# Configuración de la gráfica
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud en dB|X(f)|")
plt.title("Espectro de algunas señales en Signal")
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()

#%% Generar la ventana de Hamming
w_hamming = hamming(N).reshape(N,1)

# Graficar la ventana
# plt.figure(2)
# plt.plot(w_hamming)
# plt.title('Ventana de Hamming')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.grid(True)

SW_Hamming = Signal *w_hamming
SW_Hamming = fft(SW_Hamming , axis=0)
SW_Hamming = fftshift(SW_Hamming , axes=0)  # Centramos el espectro
SW_Hamming_norm = SW_Hamming / np.max(np.abs(SW_Hamming))

# Graficar algunas columnas
num_senales = 10
indices = np.linspace(0, N_Test-1, num_senales, dtype=int)

plt.figure(figsize=(10, 6))
for i in indices:
    plt.plot(frec, 10 * np.log10(2 * np.abs(SW_Hamming_norm[:, i])**2), label=f'Señal {i+1}')

plt.title("Espectro de señales ventaneadas con Hamming")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud en dB")
plt.grid(True)
plt.legend()

