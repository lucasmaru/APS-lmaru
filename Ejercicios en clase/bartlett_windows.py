#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:18:54 2025

@author: lmaru
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as signal

# Cerrar figuras anteriores
plt.close('all')

#%% Parámetros generales

fs = 1000                # frecuencia de muestreo (Hz)
N  = 1000                 # número de muestras
f0 = N/4                 # frecuencia a analizar (Hz)
ts = 1/fs                # tiempo de muestreo
df = fs/N
tt = np.linspace(0, (N-1)*ts, N)    #grilla temporal
ff = np.linspace(0, (N-1)*df, N)    #grilla frecuencial
bfrec = ff <= fs/2                  #vector booleano

#%% Señal senoidal y su FFT 

xx = np.sin(2 * np.pi * f0 * tt)
xx_norm = xx / np.std(xx)
ft_xx = 1/N * np.fft.fft(xx_norm)

# Gráfica
fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

# Señal en el tiempo
axs[0].plot(tt, xx_norm)
axs[0].set_title(f'Señal senoidal - f0 = {f0} Hz')
axs[0].set_xlabel('Tiempo [s]')
axs[0].set_ylabel('Amplitud')
axs[0].set_ylim([-1.2, 1.2])
axs[0].grid(True)

# Espectro
axs[1].plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_xx[bfrec])**2),
            color='orange', linestyle='dotted', label='Espectro')
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].set_ylabel('Densidad de Potencia [dB]')
axs[1].set_title('Espectro de Potencia')
axs[1].grid(True)
axs[1].legend()
plt.show()

#%% Ventana de bartlett en el tiempo y su fft

bartlett_window = signal.bartlett(N)                                #ventana bartlett
ft_bartlett = 1/N * np.fft.fft(bartlett_window)                     #calculo fft

# Eje de frecuencias centrado manualmente (sino solo grafica el lado positivo)
ff_bartlett = np.linspace(-fs/2, fs/2 - df, N)

# Reordenar la FFT para que coincida con ff_bartlett
"""
Esto es porque FFT devuelve un vector con los valores positivos primero y luego
los negativos:
    [ f = 0     →       f = 500 hz      |      f = -500hz     →       f = 0 ]
"""
ft_bartlett_acomodada = np.concatenate((ft_bartlett[N//2:], ft_bartlett[:N//2]))

# Magnitud en dB normalizada
mag_dB = 20 * np.log10(np.abs(ft_bartlett_acomodada) / np.max(np.abs(ft_bartlett_acomodada)))

# Gráficos
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(tt, bartlett_window)
plt.title("Ventana de Bartlett en el tiempo")
plt.xlabel("Tiempo [s]")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(ff_bartlett, mag_dB)
plt.title("Espectro de la ventana Bartlett")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.xlim(-500, 500)
plt.ylim(-120, 10)
plt.grid(True)

plt.tight_layout()
plt.show()
#%% Ventaneo la señal y la grafico junto con el ventaneo rectangular

window_signal = xx * bartlett_window                     #ventaneo la senoidal
window_signal_norm = window_signal/np.std(window_signal) #normalizo la señal ventaneada
ft_window_signal_norm = 1/N * np.fft.fft(window_signal_norm)  # FFT de la señal ventaneada y normalizada
plt.figure(3)
plt.plot( ff[bfrec] , 10*np.log10(2*np.abs(ft_window_signal_norm[bfrec])**2), label='Ventaneo con Bartlett', color='skyblue')
plt.plot(ff[bfrec] , 10*np.log10(2*np.abs(ft_xx[bfrec])**2), label='Ventaneo con rectangular', color='orange')
plt.xlim(0, 500)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")

#%%Conclusión
""" En N/4 tengo sintonizada la sinc y pareciera que es mejor ventanear con una fn rectangulo
pero apenas nos movemos un poco de la frecuencia de sintonia (+- 0,5 hz) el patrón se modifica.
Se ve que la señal se desparrama mucho menos con la ventana de Bartlett"""