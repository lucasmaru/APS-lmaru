#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 13:47:59 2025

@author: lmaru
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

import os

import sounddevice as sd

#%% LECTURA DEL .WAV
# Ruta base
base_path = '/home/lmaru/Documentos/UNSAM/Pendiente final/APS/Cod_clases/APS lmaru/Final/0_dB_fan/fan/id_00/'

# Archivos normal y abnormal
norm_path = os.path.join(base_path, 'normal', '00000000.wav')
abnorm_path = os.path.join(base_path, 'abnormal', '00000000.wav')

# Leer señales
fs_norm, x_norm = sio.wavfile.read(norm_path)
fs_abnorm, x_abnorm = sio.wavfile.read(abnorm_path)

# Tuplas con nombre descriptivo para legibilidad
id_00_norm = (fs_norm, x_norm)
id_00_abnorm = (fs_abnorm, x_abnorm)

# Reproducir
#sd.play(id_00_abnorm[1], id_00_abnorm[0])

#%% GRAFICO TEMPORAL CON SELECCIÓN DE CANAL

canal = 0  # Cambiar este valor para elegir otro canal (0 a 7, por ejemplo)

# Tiempo total de la señal (eje x)
tt_norm = np.arange(len(x_norm)) / fs_norm
tt_abnorm = np.arange(len(x_abnorm)) / fs_abnorm

dur = 1  # duración a graficar en segundos
N = int(dur * fs_norm)

# Graficar canal seleccionado
plt.figure(figsize=(12, 4))

plt.subplot(2,1,1)
plt.plot(tt_norm[:N], x_norm[:N, canal], label=f'Normal - canal {canal}')
plt.title(f'Señal normal - dominio temporal (canal {canal})')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.subplot(2,1,2)
plt.plot(tt_abnorm[:N], x_abnorm[:N, canal], label=f'Anómala - canal {canal}')
plt.title(f'Señal anómala - dominio temporal (canal {canal})')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.savefig("tiempo_1s.png", dpi=300)

#%% COMPARACIÓN ESPECTRAL CON WELCH

from scipy.signal import welch

# Elegir canal
canal = 0
x1 = x_norm[:, canal]
x2 = x_abnorm[:, canal]

# Estimación espectral con Welch
f1, Pxx1 = welch(x1, fs=fs_norm, window='hann', nperseg=2048)
f2, Pxx2 = welch(x2, fs=fs_abnorm, window='hann', nperseg=2048)

# Graficar
plt.figure(figsize=(12, 4))
plt.semilogy(f1, Pxx1, label='Normal')
plt.semilogy(f2, Pxx2, label='Anómala', alpha=0.8)
plt.title(f'Estimación espectral con Welch - canal {canal}')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia [V²/Hz]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(0, fs_norm//2)
plt.show()

plt.savefig("frecuencia.png", dpi=300)

