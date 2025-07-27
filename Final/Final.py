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

canal = 1  # Cambiar este valor para elegir otro canal (0 a 7, por ejemplo)

# Tiempo total de la señal (eje x)
tt_norm = np.arange(len(x_norm)) / fs_norm
tt_abnorm = np.arange(len(x_abnorm)) / fs_abnorm

dur = 10  # duración a graficar en segundos
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

#plt.savefig('SeñalTiempo_500ms.png')

#%% GRAFICO TEMPORAL PARA LOS 8 CANALES (SUBPLOTS)

dur = 0.2  # duración a graficar en segundos
N = int(dur * fs_norm)

tt_norm = np.arange(len(x_norm)) / fs_norm
tt_abnorm = np.arange(len(x_abnorm)) / fs_abnorm

for canal in range(8):
    plt.figure(figsize=(12, 4.5))
    
    # Subplot superior: señal normal
    plt.subplot(2, 1, 1)
    plt.plot(tt_norm[:N], x_norm[:N, canal], label=f'Normal - canal {canal}')
    plt.title(f'Señal normal - canal {canal}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.legend()
    
    # Subplot inferior: señal anómala
    plt.subplot(2, 1, 2)
    plt.plot(tt_abnorm[:N], x_abnorm[:N, canal], label=f'Anómala - canal {canal}')
    plt.title(f'Señal anómala - canal {canal}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

#%% GRAFICO DE LA SEÑAL PROMEDIO MULTICANAL 

dur = 0.5  # duración a graficar en segundos
N = int(dur * fs_norm)
tt = np.arange(N) / fs_norm

# Calcular el promedio multicanal
x_norm_avg = np.mean(x_norm[:N, :], axis=1)
x_abnorm_avg = np.mean(x_abnorm[:N, :], axis=1)

# Graficar
plt.figure(figsize=(12, 5))

plt.subplot(2, 1, 1)
plt.plot(tt, x_norm_avg, label='Normal - promedio 8 canales')
plt.title('Señal normal - promedio multicanal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(tt, x_abnorm_avg, label='Anómala - promedio 8 canales')
plt.title('Señal anómala - promedio multicanal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

plt.savefig('Señalmulticanal_500ms.png')

#%% GRAFICO TEMPORAL CON SELECCIÓN DE CANAL + LÍNEAS A T SEGUNDOS

canal = 0  # Cambiar este valor para elegir otro canal (0 a 7)

# Tiempo total de la señal (eje x)
tt_norm = np.arange(len(x_norm)) / fs_norm
tt_abnorm = np.arange(len(x_abnorm)) / fs_abnorm

dur = 0.2  # duración a graficar en segundos
N = int(dur * fs_norm)

# Parámetro: período estimado
T = 0.016  # segundos
lineas_t = np.arange(0.03125, dur, T)  # tiempos de las líneas

# Graficar
plt.figure(figsize=(12, 4))

# Señal normal
plt.subplot(2,1,1)
plt.plot(tt_norm[:N], x_norm[:N, canal], label=f'Normal - canal {canal}')
plt.title(f'Señal normal - dominio temporal (canal {canal})')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

# Señal anómala + líneas verticales
plt.subplot(2,1,2)
plt.plot(tt_abnorm[:N], x_abnorm[:N, canal], label=f'Anómala - canal {canal}')
for t_linea in lineas_t:
    plt.axvline(x=t_linea, color='r', linestyle='--', alpha=0.5)
plt.title(f'Señal anómala - dominio temporal (canal {canal})')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#%% COMPARACIÓN ESPECTRAL CON WELCH

from scipy.signal import welch

# Elegir canal
canal = 0
x1 = x_norm[:, canal]
x2 = x_abnorm[:, canal]

# Estimación espectral con Welch
f1, Pxx1 = welch(x1, fs=fs_norm, window='hann', nperseg=8192)
f2, Pxx2 = welch(x2, fs=fs_abnorm, window='hann', nperseg=8192)

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
f_max = 5000
plt.xlim(0, f_max)
plt.show()
#plt.savefig('Welch_5k.png')
#%% ESPECTRO DE WELCH SOBRE SEÑAL PROMEDIO MULTICANAL

# Duración total a analizar
N = int(10 * fs_norm)  # 10 segundos

# Promediar sobre los 8 canales (hasta N muestras)
x_norm_avg = np.mean(x_norm[:N, :], axis=1)
x_abnorm_avg = np.mean(x_abnorm[:N, :], axis=1)

# Calcular espectro de Welch
f_norm, Pxx_norm = welch(x_norm_avg, fs=fs_norm, nperseg=2048)
f_abnorm, Pxx_abnorm = welch(x_abnorm_avg, fs=fs_abnorm, nperseg=2048)

# Graficar en dB
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.semilogy(f_norm, Pxx_norm, label='Normal - promedio multicanal')
plt.semilogy(f_abnorm, Pxx_abnorm, label='Anómala - promedio multicanal')
plt.title('Espectro estimado por Welch (nperseg = 2048)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral [V²/Hz] (escala log)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%% AUTOCORRELACIÓN

# Parámetros
fs = 16000  # frecuencia de muestreo
dur = 0.2   # segundos que querés analizar
canal = 0

# Extraer segmento de señal anómala
N = int(dur * fs)
segmento = x_abnorm[:N, canal]  # o x_abnorm[canal][:N] si es unidimensional

# Calcular autocorrelación (modo 'full' y centrada)
corr = np.correlate(segmento, segmento, mode='full')
corr = corr[corr.size // 2:]  # quedate solo con la mitad positiva

# Normalizar
corr = corr.astype(np.float64)
corr /= np.max(corr)

# Eje de retardos (en segundos)
lags = np.arange(len(corr)) / fs

# Buscar el primer pico después de lag = 0
from scipy.signal import find_peaks
#peaks, _ = find_peaks(corr, height=0.2, distance=fs*0.01)  # descarta picos muy cerca
peaks, _ = find_peaks(corr, height=0.4, distance=fs*0.002)
if len(peaks) > 0:
    lag_muestras = peaks[0]
    freq_dominante = fs / lag_muestras
    print(f"Frecuencia dominante estimada: {freq_dominante:.2f} Hz")
else:
    print("No se detectaron picos claros en la autocorrelación.")

# Graficar
plt.figure(figsize=(10, 4))
plt.plot(lags, corr)
plt.plot(lags[peaks], corr[peaks], 'x', label='Picos')
plt.title('Autocorrelación del segmento anómalo')
plt.xlabel('Retardo [s]')
plt.ylabel('Coeficiente de correlación')
plt.grid()
plt.legend()
plt.show()

#plt.savefig('Autocorr.png')



#%% PICOS

from scipy.signal import welch, find_peaks

# Señal anómala del canal elegido
canal = 0
x2 = x_abnorm[:, canal]

# Welch
f2, Pxx2 = welch(x2, fs=fs_abnorm, window='hann', nperseg=8192)

# Encontrar picos con altura mínima
#peaks3, props = find_peaks(Pxx2, height=0.7)
peaks3, props = find_peaks(Pxx2, height=0.7, distance=fs*0.002)
# Filtrar picos dentro de 0–5000 Hz
picos_f = f2[peaks3]
picos_y = Pxx2[peaks3]
picos_filtrados = [(f, y) for f, y in zip(picos_f, picos_y) if f <= 5000]

# Graficar
plt.figure(figsize=(12, 4))
plt.semilogy(f2, Pxx2, label='Anómala - canal {}'.format(canal))
plt.title(f'Estimación espectral con Welch - canal {canal}')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia [V²/Hz]')
plt.grid(True)
plt.xlim(0, 5000)
plt.legend()

# Anotar los picos filtrados
for f_pico, y_pico in picos_filtrados:
    plt.annotate(f'{f_pico:.1f} Hz',
                 xy=(f_pico, y_pico),
                 xytext=(f_pico + 100, y_pico * 3),
                 arrowprops=dict(facecolor='red', arrowstyle='->', lw=1),
                 fontsize=9,
                 color='black')

plt.tight_layout()
plt.show()
#%% PICOS

from scipy.signal import welch

# Elegir canal
canal = 0
x1 = x_norm[:, canal]
x2 = x_abnorm[:, canal]

# Estimación espectral con Welch
f1, Pxx1 = welch(x1, fs=fs_norm, window='hann', nperseg=8192)
f2, Pxx2 = welch(x2, fs=fs_abnorm, window='hann', nperseg=8192)

# Encontrar picos con altura mínima
peaks, props = find_peaks(Pxx2, height=0.5, distance=fs*0.002)

# Filtrar picos dentro de 0–5000 Hz
picos_f = f2[peaks]
picos_y = Pxx2[peaks]
picos_filtrados = [(f, y) for f, y in zip(picos_f, picos_y) if f <= 2100]

#Filtro a mano
picos_f = [21.3 , 66.4 , 176.3 , 507.8 , 630.9, 1017.6 , 1259.8 , 1890.6 ]
picos_idx = [np.argmin(np.abs(f2 - f)) for f in picos_f]
picos_y = Pxx2[picos_idx]
picos_filtrados = list(zip(picos_f, picos_y))
#picos_y = [Pxx2[66.4], Pxx2[132.8], Pxx2[176.3], Pxx2[507.8], Pxx2[630.9], Pxx2[769.5], Pxx2[1017.6],Pxx2[1259.8],Pxx2[1890.6] ]
#picos_y = Pxx2[picos_f]

# Graficar
plt.figure(figsize=(12, 4))
plt.semilogy(f1, Pxx1, label='Normal')
plt.semilogy(f2, Pxx2, label='Anómala', alpha=0.8)
plt.title(f'Estimación espectral con Welch - canal {canal}')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia [V²/Hz]')
plt.legend()
# Anotar los picos filtrados
plt.annotate('132.8 Hz',
             xy=(132.8, 621.222),
             xytext=(150, Pxx2[picos_idx[i]] * 8),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1),
             fontsize=9,
             ha='center')
for f_pico, y_pico in picos_filtrados:
    plt.annotate(f'{f_pico:.1f} Hz',
             xy=(f_pico, y_pico),
             xytext=(f_pico , y_pico * 3),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
             fontsize=9,
             #rotation=15,  # <- inclina el texto
             color='black')

plt.grid(True)
plt.tight_layout()
f_max = 2100
plt.xlim(0, f_max)
plt.show()

#plt.savefig('Welch_flechas.png')