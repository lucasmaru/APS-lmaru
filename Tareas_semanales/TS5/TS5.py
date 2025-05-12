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
    x = x.flatten().astype(np.float64)
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

#%%##########################
# Electrocardiograma (ECG) ##
#############################

# ##Lectura##
# fs_ecg = 1000 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo
# mat_struct = sio.loadmat('./ECG_TP4.mat') #Leo el archivo
# ecg_one_lead = mat_struct['ecg_lead'].flatten() #Tomo la parte que quiero
# ecg_one_lead = ecg_one_lead[:12000] #Nos quedamos con una parte del espectro limpia pero significativa
# N_ecg = len(ecg_one_lead)

# ##Welch##
# nperseg = N_ecg // 10  #cantidad de segmentos
# noverlap = nperseg//2 #solapamiento
# f_ecg, PSD_ecg = sig.welch(ecg_one_lead, fs_ecg, window='hamming',
#                     nperseg=nperseg, noverlap=noverlap, detrend='constant')
# """
# Divido en 50 segmentos porque tengo muchos datos (aprox=1.100.000). Puedo disponer de muchos bloques 
# para bajar la varianza, sin comprometer mucho la resolución espectral porque sigo teniendo buena
# cantidad de muestras por cada bloque. Con el noverlap al 50% duplico la cantidad de bloques solapando.
# Inicialmente trabajabamos con todos los datos, luego nos quedamos con 12.000, por eso baje de 50 a 10
# segmentos. 
# """
# PSD_ecg_db = 10 * np.log10(PSD_ecg) #Paso a dB
# PSD_ecg_db_norm= PSD_ecg_db - np.max(PSD_ecg_db)#Normalizo respecto al máximo (pico en 0 dB)
# """
# Aplicamos una norma distinta a la que veniamos aplicando. En esta le resto el máximo valor a cada 
# valor de esta forma consigo que el pico de la señal quede siempre en 0dB. ¿Porque no normalizar la 
# potencia a 1 watt dividiendo por el np.std? Porque la señales que estudiamos tiene anchos de banda muy
# distintos una es de 200hz otra de 500hz y otra de 24.000hz que distorsionan la cant de frecuencias entre
# las que distrubuir ese único Watt.
# """

# ##Blackman-Tukey##
# f_ecg_bt, PSD_bt, PSD_db_bt = blackman_tukey(ecg_one_lead, fs=fs_ecg, M=N_ecg//5) #Blackman-Tukey
# PSD_db_bt_norm = PSD_db_bt - np.max(PSD_db_bt) #Normalizo respecto al máximo (pico en 0 dB)

# ##Estimación ancho de banda Welch##
# porcentaje = 0.95
# pot_total_welch =np.sum(PSD_ecg)
# pot_acumulada_welch = np.cumsum(PSD_ecg) / pot_total_welch  # ahora ya es proporción
# i = np.argmax(pot_acumulada_welch >= porcentaje)  # primer índice que supera el %
# frec_buscada_welch = f_ecg[i]

# ##Estimación ancho de banda B-T##
# pot_total_bt = np.sum(PSD_bt)
# pot_acumulada_bt = np.cumsum(PSD_bt) / pot_total_bt  # ahora ya es proporción
# i = np.argmax(pot_acumulada_bt >= porcentaje)  # primer índice que supera el %
# frec_buscada_bt = f_ecg_bt[i]

# #Visualización##
# plt.figure(figsize=(10, 6))

# plt.subplot(2, 1, 1)
# plt.plot(f_ecg, PSD_ecg_db_norm)
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Densidad espectral normalizada [dB/Hz]')
# plt.title('PSD del ECG en dB (método de Welch)')
# plt.vlines(x=frec_buscada_welch,ymin=-70,ymax=0,colors='black',linestyles='dashdot')
# plt.text(frec_buscada_welch + 5, -16, f'{frec_buscada_welch:.1f} Hz', rotation=90)
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(f_ecg_bt, PSD_db_bt_norm)
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [dB/Hz]')
# plt.title('PSD del audio en dB (método de Blackman-Tukey)')
# plt.vlines(x=frec_buscada_bt,ymin=-70,ymax=0,colors='black',linestyles='dashdot')
# plt.text(frec_buscada_welch + 5, -16, f'{frec_buscada_bt:.1f} Hz', rotation=90)
# plt.grid(True)

#%%####################
#Pletismografía (PPG)##
#######################

# """La pletismografía es una técnica para medir variaciones en el volumen de un órgano o parte del cuerpo, 
# generalmente relacionadas con el flujo sanguíneo (oximetro de pulso). Cuando hablamos de PPG (sigla de 
# Photoplethysmography), nos referimos a una técnica óptica que mide cambios en el volumen sanguíneo en 
# los tejidos."""

# ##Lectura##
# fs_ppg = 400 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo
# ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe
# N_ppg = len(ppg)

# ##Welch##
# nperseg = N_ppg // 5  #cantidad de segmentos
# noverlap = nperseg//2 #solapamiento
# f_ppg, PSD_ppg = sig.welch(ppg, fs_ppg, window='hamming',
#                     nperseg=nperseg, noverlap=noverlap, detrend= 'linear')
# PSD_ppg_db = 10 * np.log10(PSD_ppg) #Paso a dB
# PSD_ppg_db_norm = PSD_ppg_db - np.max(PSD_ppg_db) #Normalizo respecto al máximo (pico en 0 dB)

# ##Blackman-Tukey##
# f_ppg_bt, PSD_bt, PSD_db_bt = blackman_tukey(ppg, fs=fs_ppg) #Blackman-Tukey
# PSD_db_bt_norm = PSD_db_bt - np.max(PSD_db_bt) #Normalizo respecto al máximo (pico en 0 dB)

# ##Estimación ancho de banda Welch##
# porcentaje = 0.95
# pot_total_welch =np.sum(PSD_ppg)
# pot_acumulada_welch = np.cumsum(PSD_ppg) / pot_total_welch  # ahora ya es proporción
# i = np.argmax(pot_acumulada_welch >= porcentaje)  # primer índice que supera el %
# frec_buscada_welch = f_ppg[i]

# ##Estimación ancho de banda B-T##
# pot_total_bt = np.sum(PSD_bt)
# pot_acumulada_bt = np.cumsum(PSD_bt) / pot_total_bt  # ahora ya es proporción
# i = np.argmax(pot_acumulada_bt >= porcentaje)  # primer índice que supera el %
# frec_buscada_bt = f_ppg_bt[i]

# plt.figure(figsize=(10, 6))

# plt.subplot(2, 1, 1)
# plt.plot(f_ppg, PSD_ppg_db_norm)
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Densidad espectral normalizada [dB/Hz]')
# plt.title('PSD del PPG en dB (método de Welch)')
# plt.vlines(x=frec_buscada_welch,ymin=-70,ymax=0,colors='black',linestyles='dashdot')
# plt.text(frec_buscada_welch + 5, -16, f'{frec_buscada_welch:.1f} Hz', rotation=90)
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(f_ppg_bt, PSD_db_bt_norm)
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [dB/Hz]')
# plt.title('PSD del PPG en dB (método de Blackman-Tukey)')
# plt.vlines(x=frec_buscada_bt,ymin=-70,ymax=0,colors='black',linestyles='dashdot')
# plt.text(frec_buscada_welch + 5, -16, f'{frec_buscada_bt:.1f} Hz', rotation=90)
# plt.grid(True)

#%%#################
# Lectura de audio #
####################

##Lectura##
#fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav') #esta funciòn devuelve la fs
fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
#fs_audio, wav_data = sio.wavfile.read('silbido.wav')
N_audio = len(wav_data)

##Welch##
nperseg = N_audio // 5  #cantidad de segmentos
noverlap = nperseg//2 #solapamiento
f_audio, PSD_audio = sig.welch(wav_data, fs_audio, window='hamming',
                    nperseg=nperseg, noverlap=noverlap, detrend= 'linear')
PSD_audio_db = 10 * np.log10(PSD_audio) #Paso a dB
PSD_audio_db_norm = PSD_audio_db - np.max(PSD_audio_db) #Normalizo respecto al máximo (pico en 0 dB)

##Blackman-Tukey##
f_bt_audio, PSD_bt, PSD_db_bt = blackman_tukey(wav_data, fs=fs_audio)
PSD_db_bt_norm = PSD_db_bt - np.max(PSD_db_bt) #Normalizo respecto al máximo (pico en 0 dB)

##Estimación ancho de banda Welch##
porcentaje = 0.98
pot_total_welch =np.sum(PSD_audio)
pot_acumulada_welch = np.cumsum(PSD_audio) / pot_total_welch  # ahora ya es proporción
i = np.argmax(pot_acumulada_welch >= porcentaje)  # primer índice que supera el %
frec_buscada_welch = f_audio[i]

##Estimación ancho de banda B-T##
pot_total_bt = np.sum(PSD_bt)
pot_acumulada_bt = np.cumsum(PSD_bt) / pot_total_bt  # ahora ya es proporción
i = np.argmax(pot_acumulada_bt >= porcentaje)  # primer índice que supera el %
frec_buscada_bt = f_bt_audio[i]

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(f_audio, PSD_audio_db_norm)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral normalizada [dB/Hz]')
plt.title('PSD del audio en dB (método de Welch)')
plt.vlines(x=frec_buscada_welch,ymin=-90,ymax=0,colors='black',linestyles='dashdot',label=f'BW {porcentaje *100:.0f}% {frec_buscada_welch:.2f} hz')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(f_bt_audio, PSD_db_bt_norm)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('PSD del audio en dB (método de Blackman-Tukey)')
plt.vlines(x=frec_buscada_bt,ymin=-90,ymax=0,colors='black',linestyles='dashdot',label=f'BW {porcentaje *100:.0f}% {frec_buscada_welch:.2f} hz')
plt.legend()
plt.grid(True)