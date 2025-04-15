#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 11:47:42 2025

@author: lmaru
"""

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
f0 = N/4+0.5                 # frecuencia a analizar (Hz)
"""sintonizo la fo en mitad de banda digital que va de 0 a Niquist (fs/2) esto
es importante conceptualmente porque en este punto estamos lejos de los efectos
de borde que nos propon el enfoque basado en ventanas """
ts = 1/fs                # tiempo de muestreo
df = fs/N
tt = np.linspace(0, (N-1)*ts, N)    #grilla temporal
ff = np.linspace(0, (N-1)*df, N)    #grilla frecuencial
bfrec = ff <= fs/2                  #vector booleano

#%% Señal senoidal y su FFT 

xx = np.sin(2 * np.pi * f0 * tt)
xx_norm = xx / np.std(xx) #normalizo dividiendo por el desvío standar
ft_xx = 1/N * np.fft.fft(xx_norm)

# Gráfica
# fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

# # Señal en el tiempo
# axs[0].plot(tt, xx_norm)
# axs[0].set_title(f'Señal senoidal - f0 = {f0} Hz')
# axs[0].set_xlabel('Tiempo [s]')
# axs[0].set_ylabel('Amplitud')
# axs[0].set_ylim([-1.2, 1.2])
# axs[0].grid(True)

# # Espectro
# axs[1].plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_xx[bfrec])**2),
#             color='orange', linestyle='dotted', label='Espectro')
# axs[1].set_xlabel('Frecuencia [Hz]')
# axs[1].set_ylabel('Densidad de Potencia [dB]')
# axs[1].set_title('Espectro de Potencia')
# axs[1].grid(True)
# axs[1].legend()
# plt.show()

#%% Ventana de blackman en el tiempo y su fft

blackman_window = signal.blackman(N)                                #ventana blackman
ft_blackman = 1/N * np.fft.fft(blackman_window)                     #calculo fft

# Eje de frecuencias centrado manualmente (sino solo grafica el lado positivo)
ff_blackman = np.linspace(-fs/2, fs/2 - df, N)

# Reordenar la FFT para que coincida con ff_blackman
"""
Esto es porque FFT devuelve un vector con los valores positivos primero y luego
los negativos:
    [ f = 0     →       f = 500 hz      |      f = -500hz     →       f = 0 ]
"""
ft_blackman_acomodada = np.concatenate((ft_blackman[N//2:], ft_blackman[:N//2]))

# Magnitud en dB normalizada
mag_dB = 20 * np.log10(np.abs(ft_blackman_acomodada) / np.max(np.abs(ft_blackman_acomodada)))

# Gráficos
# plt.figure(2)
# plt.subplot(2,1,1)
# plt.plot(tt, blackman_window)
# plt.title("Ventana de blackman en el tiempo")
# plt.xlabel("Tiempo [s]")
# plt.grid(True)

# plt.subplot(2,1,2)
# plt.plot(ff_blackman, mag_dB)
# plt.title("Espectro de la ventana blackman")
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("Magnitud [dB]")
# plt.xlim(-500, 500)
# plt.ylim(-120, 10)
# plt.grid(True)

# plt.tight_layout()
# plt.show()
#%% Ventaneo la señal y la grafico junto con el ventaneo rectangular

window_signal = xx * blackman_window                     #ventaneo la senoidal
window_signal_norm = window_signal/np.std(window_signal) #normalizo la señal ventaneada
ft_window_signal_norm = 1/N * np.fft.fft(window_signal_norm)  # FFT de la señal ventaneada y normalizada
plt.figure(3)
plt.plot( ff[bfrec] , 10*np.log10(2*np.abs(ft_window_signal_norm[bfrec])**2), label='Ventaneo con blackman', color='skyblue')
plt.plot(ff[bfrec] , 10*np.log10(2*np.abs(ft_xx[bfrec])**2), label='Ventaneo con rectangular', color='red')
plt.xlim(0, 500)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.legend()

#%%Conclusión
""" En N/4 tengo sintonizada la sinc y pareciera que es mejor ventanear con una fn rectangulo
pero apenas nos movemos un poco de la frecuencia de sintonia (+- 0,5 hz) el patrón se modifica.
Se ve que la señal se desparrama mucho menos con la ventana de blackman"""