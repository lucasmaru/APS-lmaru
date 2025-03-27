#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 20:34:02 2025

@author: lmaru
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 20:19:37 2025

@author: lmaru
"""

import numpy as np
import matplotlib.pyplot as plt

# Cerrar figuras anteriores
plt.close('all')

# Parámetros generales
f0_vec = [0.5, 500, 1000]  # 3 frecuencias distintas
fs = 2000                # frecuencia de muestreo (Hz)
N = 1000                 # número de muestras
ts = 1/fs
df = fs/N
tt = np.linspace(0, (N-1)*ts, N)
ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs/2

# Loop sobre cada frecuencia
for i, f0 in enumerate(f0_vec):
    xx = np.sin(2 * np.pi * f0 * tt)
    ft_xx = 1/N * np.fft.fft(xx)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # Señal en el tiempo
    axs[0].plot(tt, xx)
    axs[0].set_title(f'Señal senoidal - f0 = {f0} Hz')
    axs[0].set_xlabel('Tiempo [s]')
    axs[0].set_ylabel('Amplitud')
    axs[0].set_ylim([-1.2, 1.2])  # rango fijo del eje Y
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
