#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 15:21:25 2025

@author: lmaru
"""
#%%IMPORTACIONES
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio
#%%APROXIMANTES Y TIPO DE FILTRO

aprox_name = 'butter'
#aprox_name = 'cheby1'
#aprox_name = 'cheby2'
#aprox_name = 'ellip'

filter_type = 'bandpass'

if filter_type == 'lowpass':

    # fpass = 1/2/np.pi #
    fpass = 0.25
    ripple = 0.5  # dB
    fstop = 0.6  # Hz
    attenuation = 40  # dB

elif filter_type == 'bandpass':

    fpass = np.array([1.0, 35.0])
    ripple = 1  # dB
    fstop = np.array([.1, 50.])
    attenuation = 40  # dB

#%% DISEÑO Y TESTEO
fs = 1000
mi_sos = sig.iirdesign(wp=fpass, ws=fstop, gpass=ripple, gstop=attenuation,
                       analog=False, ftype=aprox_name, output='sos', fs=fs)
"""
mi_sos es una matriz de 14 filas por tanto el filtro diseñado es de orden 28. Cada fila representa
un biquad de segundo orden, cada fila contiene el valor de los coef b0, b1, b2, a0, a1 y a2. Por eso 
es de 6 X 14
"""

npoints = 1000

nyq_frec = fs/2
wrad = np.append(np.logspace(-2, 0.8, 250) , np.logspace(0.9, 1.6, 250))
wrad = np.append(wrad , np.linspace(40, nyq_frec,500, endpoint=True)) / nyq_frec
"""
A sosfreqz le podemos pasar un entero y contruye una grilla lineal de npoints, o le podemos pasar una grilla
de frec con espaciamiento logarítmico creada especialemnte para ver con mejor resolución el codo estrecho
entre 0 y 0,1 hz
"""

w, hh = sig.sosfreqz(mi_sos, wrad)

"""
sosfreqz levanta la respuesta en frecuencia del filtro, calcula módulo y fase y hace un barrido númerico
de nponits puntos de 0 a pi, es decir un barrido de frecuencia con 1000 valores entre 0 y pi. En hh nos
llevamos un vector de complejos donde esta el módulo y la fase y w es el que va a hacer de eje x. 
"""
plt.plot(w/np.pi*fs/2, 20*np.log10(np.abs(hh)), label='mi_sos')
"""
El gráfico va de 0 a pi, lo divide por pi para llevarlo de 0 a 1 y por último multiplica por fs/s 
para llevarlo de 0 a Niquist 
"""
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plot_plantilla(filter_type=filter_type, fpass=fpass, ripple=ripple,
               fstop=fstop, attenuation=attenuation, fs=fs)
plt.legend()

#%% LECTURA DE DATOS Y FILTRADO DEL ECG
fs_ecg = 1000 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo
mat_struct = sio.loadmat('./ECG_TP4.mat') #Leo el archivo
ECG = mat_struct['ecg_lead'].flatten() #Tomo la parte que quiero
#ecg_one_lead = ecg_one_lead[:12000] #Nos quedamos con una parte del espectro limpia pero significativa
N_ECG = len(ECG)
ECG_filt = sig.sosfiltfilt(mi_sos,ECG)

#%%VISUALIZACIÒN
t = np.arange(N_ECG) / fs_ecg  # Vector de tiempo

plt.figure(figsize=(12, 6))

#plt.subplot(2, 1, 1)
plt.plot(t[1200:], ECG[1200:], label='ECG original')
plt.title('Señal ECG Original')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

#plt.subplot(2, 1, 2)
plt.plot(t[1200:], ECG_filt[1200:], label='ECG filtrada', color='orange')
plt.title('Señal ECG Filtrada (Butterworth pasa banda)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
