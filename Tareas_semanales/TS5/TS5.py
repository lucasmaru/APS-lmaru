#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 21:46:19 2025

@author: lmaru
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio

#%% Función Blackman-Tukey
def blackman_tukey(x, fs, M=None):
    x = x.flatten()
    N = len(x)
    
    if M is None:
        M = N // 5
    
    r_len = 2 * M - 1
    xx = x[:r_len]
    
    r = np.correlate(xx, xx, mode='same') / r_len
    w = sig.windows.blackman(r_len)
    r_win = r * w

    PSD = np.abs(np.fft.fft(r_win, n=N))
    PSD_db = 10 * np.log10(np.maximum(PSD, 1e-12))  # para evitar log(0)
    
    f = np.fft.fftfreq(N, d=1/fs)
    pos = f >= 0  # máscara para parte positiva

    return f[pos], PSD[pos], PSD_db[pos]

#%%###############
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')

mat_struct = sio.loadmat('./ECG_TP4.mat') #Leo el archivo

ecg_one_lead = mat_struct['ecg_lead'].flatten()
N_ecg = len(ecg_one_lead)

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])

ecg_potencia_unitaria = ecg_one_lead /np.std(ecg_one_lead) #normalizo en potencia

nperseg = N_ecg // 50  #cantidad de segmentos
noverlap = nperseg//2 #solapamiento
f_ecg, PSD_ecg = sig.welch(ecg_potencia_unitaria, fs_ecg, window='hamming',
                    nperseg=nperseg, noverlap=noverlap, detrend='constant')

"""
Divido en 50 segmentos porque tengo muchos datos (aprox=1.100.000). Puedo disponer de muchos bloques 
para bajar la varianza, sin comprometer mucho la resolución espectral porque sigo teniendo buena
cantidad de muestras por cada bloque. Con el noverlap al 50% duplico la cantidad de bloques solapando. 
"""
PSD_ecg_db = 10 * np.log10(PSD_ecg) #Paso a dB
PSD_ecg_db_norm= PSD_ecg_db - np.max(PSD_ecg_db)#Normalizo respecto al máximo (pico en 0 dB)
"""
Aplicamos una norma distinta a la que veniamos aplicando. En esta le resto el máximo valor a cada 
valor de esta forma consigo que el pico de la señal quede siempre en 0dB. ¿Porque no normalizar la 
potencia a 1 watt dividiendo por el np.std? Porque la señales que estudiamos tiene anchos de banda muy
distintos una es de 200hz otra de 500hz y otra de 24.000hz. Pero como también me sirve que la potencia
sea unitaria para estimar luego el ancho de banda, aplico ambas normalizaciones. Entonces obtengo una
potencia normalizada y con los picos del espectro en 0dB, es decir relativos al máximo del espectro.
"""
f_ecg_bt, PSD_bt, PSD_db_bt = blackman_tukey(ecg_potencia_unitaria, fs=fs_ecg, M=N_ecg//5) #Blackman-Tukey
PSD_db_bt_norm = PSD_db_bt - np.max(PSD_db_bt) #Normalizo respecto al máximo (pico en 0 dB)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(f_ecg, PSD_ecg_db_norm)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral normalizada [dB/Hz]')
plt.title('PSD del ECG en dB (método de Welch)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(f_ecg_bt, PSD_db_bt_norm)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('PSD del audio en dB (método de Blackman-Tukey)')
plt.grid(True)


# #%%#################################
# # Lectura de pletismografía (PPG)  #
# ####################################

# """La pletismografía es una técnica para medir variaciones en el volumen de un órgano o parte del cuerpo, 
# generalmente relacionadas con el flujo sanguíneo (oximetro de pulso). Cuando hablamos de PPG (sigla de 
# Photoplethysmography), nos referimos a una técnica óptica que mide cambios en el volumen sanguíneo en 
# los tejidos."""

# fs_ppg = 400 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo

# # Cargar el archivo CSV como un array de NumPy
# ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe
# # plt.figure()
# # plt.plot(ppg)

# N_ppg = len(ppg)
# ppg_potencia_unitaria = ppg /np.std(ppg) #normalizo en potencia

# nperseg = N_ppg // 5  #cantidad de segmentos
# noverlap = nperseg//2 #solapamiento
# f_ppg, PSD_ppg = sig.welch(ppg_potencia_unitaria, fs_ppg, window='hamming',
#                     nperseg=nperseg, noverlap=noverlap, detrend= 'linear')

# PSD_ppg_db = 10 * np.log10(PSD_ppg) #Paso a dB
# PSD_ppg_db_norm = PSD_ppg_db - np.max(PSD_ppg_db) #Normalizo respecto al máximo (pico en 0 dB)

# f_ppg_bt, PSD_bt, PSD_db_bt = blackman_tukey(ppg_potencia_unitaria, fs=fs_ppg) #Blackman-Tukey
# PSD_db_bt_norm = PSD_db_bt - np.max(PSD_db_bt) #Normalizo respecto al máximo (pico en 0 dB)

# plt.figure(figsize=(10, 6))

# plt.subplot(2, 1, 1)
# plt.plot(f_ppg, PSD_ppg_db_norm)
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Densidad espectral normalizada [dB/Hz]')
# plt.title('PSD del PPG en dB (método de Welch)')
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(f_ppg_bt, PSD_db_bt_norm)
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [dB/Hz]')
# plt.title('PSD del PPG en dB (método de Blackman-Tukey)')
# plt.grid(True)


# #%%#################
# # Lectura de audio #
# ####################

# # Cargar el archivo CSV como un array de NumPy
# fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav') #esta funciòn devuelve la fs
# #fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# #fs_audio, wav_data = sio.wavfile.read('silbido.wav')

# # plt.figure()
# # plt.plot(wav_data)

# N_audio = len(wav_data)
# wav_data_potencia_unitaria = wav_data /np.std(wav_data) #normalizo en potencia

# nperseg = N_audio // 5  #cantidad de segmentos
# noverlap = nperseg//2 #solapamiento
# f_audio, PSD_audio = sig.welch(wav_data_potencia_unitaria, fs_audio, window='hamming',
#                     nperseg=nperseg, noverlap=noverlap, detrend= 'linear')

# PSD_audio_db = 10 * np.log10(PSD_audio) #Paso a dB
# PSD_audio_db_norm = PSD_audio_db - np.max(PSD_audio_db) #Normalizo respecto al máximo (pico en 0 dB)

# f_bt_audio, PSD_bt, PSD_db_bt = blackman_tukey(wav_data_potencia_unitaria, fs=fs_audio)
# PSD_db_bt_norm = PSD_db_bt - np.max(PSD_db_bt) #Normalizo respecto al máximo (pico en 0 dB)


# plt.figure(figsize=(10, 6))

# plt.subplot(2, 1, 1)
# plt.plot(f_audio, PSD_audio_db_norm)
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Densidad espectral normalizada [dB/Hz]')
# plt.title('PSD del audio en dB (método de Welch)')
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(f_bt_audio, PSD_db_bt_norm)
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [dB/Hz]')
# plt.title('PSD del audio en dB (método de Blackman-Tukey)')
# plt.grid(True)