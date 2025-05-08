#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:55:30 2023

@author: mariano
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

#%%
def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

#%%###############
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N = len(ecg_one_lead)

hb_1 = vertical_flaten(mat_struct['heartbeat_pattern1'])
hb_2 = vertical_flaten(mat_struct['heartbeat_pattern2'])

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])

# # un patrón promedio o típico de un tipo de latido.
# plt.figure()
# plt.plot(hb_1)

# # un patrón de latido ectópico que puede estar asociado a anomalìas cardiacas
# plt.figure()
# plt.plot(hb_2)

nperseg = N // 20
noverlap=nperseg//2
f_ecg, PSD_ecg = sig.welch(ecg_one_lead, fs_ecg, window='hann', nperseg=nperseg, noverlap=noverlap)

"""
Divido en 20 segmentos porque tengo muchos datos (aprox=1.100.000). Puedo disponer de muchos bloques 
para bajar la varianza, sin comprometer mucho la resolución espectral porque sigo teniendo buena
cantidad de muestras por cada bloque. Con el noverlap al 50% duplico la cantidad de bloques solapando. 
"""
plt.figure()
plt.plot(f_ecg, 10 * np.log10(PSD_ecg))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral [dB/Hz]')
plt.title('PSD del ECG en dB (método de Welch)')
plt.grid(True)


#%%#################################
# # Lectura de pletismografía (PPG)  #
# ####################################

# """La pletismografía es una técnica para medir variaciones en el volumen de un órgano o parte del cuerpo, 
# generalmente relacionadas con el flujo sanguíneo (oximetro de pulso). Cuando hablamos de PPG (sigla de 
# Photoplethysmography), nos referimos a una técnica óptica que mide cambios en el volumen sanguíneo en 
# los tejidos."""

# fs_ppg = 400 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo

# # Cargar el archivo CSV como un array de NumPy
# ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

# plt.figure()
# plt.plot(ppg)

#%%#################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
#fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
#fs_audio, wav_data = sio.wavfile.read('silbido.wav')

plt.figure()
plt.plot(wav_data)

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)
#%%
