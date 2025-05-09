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
from scipy.io.wavfile import write

#%%###############
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])

# # un patrón promedio o típico de un tipo de latido.
# plt.figure()
# plt.plot(hb_1)

# # un patrón de latido ectópico que puede estar asociado a anomalìas cardiacas
# plt.figure()
# plt.plot(hb_2)

ecg_potencia_unitaria = ecg_one_lead /np.std(ecg_one_lead) #normalizo en potencia

nperseg = N // 50
noverlap=nperseg//2
f_ecg, PSD_ecg = sig.welch(ecg_potencia_unitaria, fs_ecg, window='hamming',
                           nperseg=nperseg, noverlap=noverlap, detrend= 'linear')

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
distintos una es de 200hz otra de 500hz y otra de 24.000hz
"""
plt.figure()
plt.plot(f_ecg, PSD_ecg_db_norm)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral normalizada [dB/Hz]')
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

#plt.figure()
#plt.plot(wav_data)

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)
#%%
