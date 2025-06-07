#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:13:06 2025

@author: lmaru
"""

#%%IMPORTACIONES
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio
#%%APROXIMANTES Y TIPO DE FILTRO

filter_type = 'bandpass'

fpass = np.array([1.0, 35.0])
ripple = 1  # dB
fstop = np.array([.1, 50.])
attenuation = 40  # dB
#%% DISEÑO Y TESTEO
fs = 1000
cant_coef = 25002 #Orden = cant_coef - 1, el orden impar
window ='hamming'


freq = [0, 0.1, 1.0, 35.0, 50.0, fs/2]
gain = [0, 0, 1, 1, 0, 0]

mi_fir = sig.firwin2(numtaps=cant_coef, freq=freq, gain=gain, fs=fs, window='hamming')


"""
mi_sos es una matriz de 15 filas por tanto el filtro diseñado es de orden 30. Los FIR necesitan más orden
para lograr lo mismo, empiezo con 101 coef, es decir orden 100.Luego vamos tanteando.
"""

npoints=8000
w, hh = sig.freqz(mi_fir, worN=npoints, fs=fs) #...fs=fs=>devuelve w en hz directamente

"""
freqz levanta la respuesta en frecuencia del filtro, calcula módulo y fase y hace un barrido númerico
de nponits puntos de 0 a pi, es decir un barrido de frecuencia con 1000 valores entre 0 y pi. En hh nos
llevamos un vector de complejos donde esta el módulo y la fase y w es el que va a hacer de eje x. 
"""
plt.figure(1)
plt.subplot(2,1,1)

plt.plot(w, 20*np.log10(np.abs(hh)), label=f'FIR con-{window} (orden {cant_coef - 1})')
"""
En este plot puedo usar w directo porque ya esta en hz, porque a freqz le indicamos ...fs=fs que devuelve
directo a w en hz en lugar de radianes por muestre es decir ya hace la conversión que en el IIR hicimos con la 
la linea w/np.pi*fs/2 para pasar de 0 a pi a 0 y Niquist
"""
plt.title('Plantilla de diseño (respuesta de módulo)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plot_plantilla(filter_type=filter_type, fpass=fpass, ripple=ripple,
                fstop=fstop, attenuation=attenuation, fs=fs)
plt.legend()

plt.subplot(2,1,2)
fase = np.unwrap(np.angle(hh))#unwrap desenrrolla la fase quita las discontinuidades

demora = -np.diff(fase) / np.diff(w/np.pi*fs/2)
w_med = (w[1:] + w[:-1]) / 2  # promedio entre cada par de muestras de w
plt.plot(w, fase, label='Fase') 
plt.plot(w_med, demora, label='Demora') 
plt.xlabel('Frecuencia[Hz]')
plt.ylabel('Fase`[rad]')
plt.legend()

#%% LECTURA DE DATOS Y FILTRADO DEL ECG
fs_ecg = 1000 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo
mat_struct = sio.loadmat('./ECG_TP4.mat') #Leo el archivo
ECG = mat_struct['ecg_lead'].flatten() #Tomo la parte que quiero
#ecg_one_lead = ecg_one_lead[:12000] #Nos quedamos con una parte del espectro limpia pero significativa
N_ECG = len(ECG)
ECG_filt = sig.lfilter(mi_fir, 1, ECG)


#%%VISUALIZACIÒN
# t = np.arange(N_ECG) / fs_ecg  # Vector de tiempo

# plt.figure(2,figsize=(12, 6))

# #plt.subplot(2, 1, 1)
# plt.plot(t, ECG, label='ECG original')
# plt.title('Señal ECG Original')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.grid(True)
# plt.legend()

# #plt.subplot(2, 1, 2)
# plt.plot(t, ECG_filt, label='ECG filtrada', color='orange')
# plt.title(f'Señal ECG Filtrada (firwin con {window} orden {cant_coef - 1})')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.grid(True)
# plt.legend()

# plt.tight_layout()
# plt.show()