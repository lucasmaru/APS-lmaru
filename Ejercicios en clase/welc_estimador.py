#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 19:14:15 2025

@author: lmaru
"""

#%% módulos y funciones a importar
from scipy.signal import welch
import numpy as np
np.random.seed(52)  # Fijamos la semilla para resultados reproducibles
from scipy.signal.windows import hamming, hann, blackmanharris, flattop
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

mostrar_graficos = False
#%% Datos de la simulacion

fs = 1000.0           # frecuencia de muestreo (Hz)
N = 1000            # cantidad de muestras
ts = 1/fs           # tiempo de muestreo
df = fs/N           # resolución espectral

N_Test = 200        # Numero de pruebas

SNR = 10            # Signal to Noise Ratio 
 
"""De la definición de SNR y asumiendo potencia de la señal normalizada llego 
al valor que debe tener la potencia de ruido para respetar el SNR requerido.
De esta manera tengo resuelta la experiencia para ambas SNR prescriptas."""
Sigma2 = 10**(-10/SNR) #Potencia de ruido

Omega_0 = fs/4      # Nos ponemos a mitad de banda digital
"""Esto viene prescripto en el enunciado y tiene la intención de generar
frecuencias que oscilen +- medio bin del centro de banda digital."""

#%% Genero mi matriz de 1000x200 de la senoidal

"""Defino la amplitud que ya calculamos en otras TS para que la potencia de las
200 senoidales quede normalizada"""
A1 = np.sqrt(2) 

"""Genero el vector de 1x200 frecuencias, 200 valores extraidos de una 
distribución uniforme de -1,2 a 1/2 y le fuerzo las dimensiones con reshape"""
fr = np.random.uniform(-1/2,1/2,N_Test).reshape(1,N_Test)
  
Omega_1 = Omega_0 + fr*df                       # Genero mi Omega_1 de 1x200

# Genero vector de tiempo para meterlo como mi matriz de 1000x200 en el seno 
tt = np.linspace(0, (N-1)*ts, N).reshape(N,1)    #vector columna de 1000x1
tt = np.tile(tt, (1, N_Test))  # tile repite esa columna 200 veces, queda de 1000x200

""" Al mutiplicar omega_1 con tt numpy por defecto multiplica término a término
como tt es de 1000x200 entiende que tiene que expandir dimensionalmente a 
omega_1 para poder hacer el producto término a término, lo hace automáticamente
y por eso S es de 1000x200 
"""
S = A1 * np.sin(2 * np.pi * Omega_1 * tt)

"""Grafico la columna 0 del tiempo y la columna cero de S, para corroborar que 
tengo una senoidal pura de una frecuenia de alrededor de 250hz en cada columa, 
pero en cambio veo algo como una envolvente que módula la señal,pero si pongo 
omega_0=1 veo lo que espero ver"""

#%% Genero el ruido para la señal
# Para poder general la señal de ruido, tenemos que tener una distribucion normal con un N(o,sigma)

Media = 0                   # Media
SD_Sigma = np.sqrt(Sigma2)  # Desvio standar a partir de la pot calculada antes 

nn = np.random.normal(Media, SD_Sigma, N).reshape(N,1)  # Genero señal de ruido 1000x1
nn = np.tile(nn, (1,N_Test))                            # tile repite esa columna 200 veces, queda de 1000x200

#%% Sumo la matriz de senoidales con el ruido
Signal = S + nn
"""Grafico la columna 0 del tiempo y la columna cero de Signal, para corroborar 
que se haya añadido el ruido. Se ve el ruido, pero con el mismo patrón que cuando 
grafique la senoidal pura, se ve como una envolvente que módula la señal, 
nuevamente si cambio a omega_0=1 veo algo más razonable"""

###########################GRAFICO DE CHEQUEO#################################
# if mostrar_graficos:
#     plt.figure(1)
#     plt.plot(tt[:,0], Signal[:,0:1])  # ahora sí, una senoidal limpia
#     plt.xlabel("Tiempo [s]")
#     plt.ylabel("Amplitud")
#     plt.title("Senoidal + ruido")
#     plt.grid(True)

#%% Visualización: Espectros de Welch de algunas señales
nperseg=N//4

ventana_welch = 'hamming'  # podés cambiar a 'hann', 'flattop', etc.

f, Pxx = welch(Signal, detrend ='linear', nfft= None,
               average ='median', fs=fs, window = ventana_welch, 
               nperseg= nperseg, scaling='density', return_onesided=True, 
               noverlap = nperseg//2, axis=0)
if mostrar_graficos:
    plt.figure(figsize=(10, 6))
    plt.plot(f, 10 * np.log10(Pxx))
    plt.title(f"Espectro por Welch (ventana: {ventana_welch})")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Densidad espectral de potencia [dB]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%% Estimador de amplitud usando Welch

# Array para guardar el estimador de amplitud para cada señal
a1_hat_welch = np.zeros(N_Test)
idx_max = np.argmax(Pxx)              # Buscamos el máximo en el espectro estimado
#a1_hat_welch = Pxx[idx_max]

# Visualización del estimador de amplitud por Welch
#if mostrar_graficos:
plt.figure(figsize=(8,5))
plt.hist(a1_hat_welch, bins=30, alpha=0.7, edgecolor='black')
plt.axvline(np.mean(a1_hat_welch), color='blue', linestyle='--', label=f"Media: {np.mean(a1_hat_welch):.2f}")
plt.axvline(A1, color='r', linestyle='--', label=r"Valor real $a_1 = \sqrt{2}$")
plt.title("Estimador de amplitud por Welch (Hamming)")
plt.xlabel("Amplitud estimada")
plt.ylabel("Ocurrencias por bin")
plt.grid(True)
plt.legend()
plt.show()