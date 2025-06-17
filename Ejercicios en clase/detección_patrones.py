#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 15:42:52 2025

@author: lmaru
"""

#%%IMPORTACIONES
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio

plt.close('all')
#%%ECG
fs_ecg = 1000 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo
mat_struct = sio.loadmat('./ECG_TP4.mat') #Leo el archivo
ecg_one_lead = mat_struct['ecg_lead'].flatten() #Tomo la parte que quiero
N_ecg = len(ecg_one_lead)
qrs = mat_struct['qrs_pattern1'].flatten()

#%% Normalizado y correlación
ecg_one_lead = ecg_one_lead/np.std(ecg_one_lead)
qrs = qrs/np.std(qrs)

correlacion = np.correlate(ecg_one_lead,qrs,'same')
"""mode='same' Por qué devuelve un vector del mismo largo que la señal de ECG, lo cual facilita el 
alineamiento temporal con la señal original; permite aplicar umbrales directamente sobre el 
resultado para detectar eventos (picos altos ≈ coincidencia con la plantilla).
"""
correlacion = correlacion/np.std(correlacion) #Normalizo la salida

retardo = len(qrs) // 2
# Duración mínima entre latidos en segundos
fs = 1000  # frecuencia de muestreo en Hz
dist_min = int(0.3 * fs)  # 300 ms de separación entre latidos
# Detección de picos
picos,_ = sig.find_peaks(correlacion, height=0.15, distance=dist_min) 

# Eje temporal
t_ecg = np.arange(N_ecg) / fs_ecg
GRAFICAR_1= False
if GRAFICAR_1:
     # Dominio temporal
    plt.figure(figsize=(12, 4))
    plt.plot(t_ecg, correlacion,color='orange',label='Correlación')
    plt.plot(t_ecg[picos], correlacion[picos],'rx',
             label=f'Picos con find_peaks (n={len(picos)})')
    plt.plot(t_ecg, ecg_one_lead, label='ECG original' )
    plt.title("ECG original")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
"""En la grafico vemos que detecta bastante bien los picos e introduce una pequeña demora. El
ruido de alta frecuencia pasa absolutmente asique decimos que del punto del filtrado tiene un
comportamiento bastante pasabajos. Podemos conseguir mejores resultados si la señal esta pre-
filtrada"""
#%% Repito con la señal filtrada
ecg_filtrado = np.load("ecg_iir_cheby2.npy")  # solo una
ecg_filtrado = ecg_filtrado/np.max(np.abs(ecg_filtrado))
correlacion_filt = np.correlate(ecg_filtrado,qrs,'same')
#correlacion_filt = correlacion_filt/np.std(correlacion_filt) #Normalizo la salida

correlacion_filt = correlacion_filt / np.max(np.abs(correlacion_filt))
umbral = 0.15  # ahora es 60% de 1


picos_filt,_ = sig.find_peaks(correlacion_filt, height=umbral, distance=dist_min)
latidos = mat_struct['qrs_detections'].flatten()
GRAFICAR_2= True
if GRAFICAR_2:
    plt.figure(figsize=(12, 4))
    plt.plot(t_ecg, correlacion_filt,color='orange',label='Correlación')
    plt.plot(t_ecg[picos_filt], correlacion_filt[picos_filt],'rx',
             label=f'Picos con find_peaks (n={len(picos_filt)})')
    plt.plot(t_ecg, ecg_filtrado, label='ECG filtrado', alpha=0.5 )
    plt.title("ECG prefiltrado")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
"""Con la señal filtrada y normalizando respecto al maximo pude encontrar 1903 picos que son los mismos que
los que hay en mat_struct.
"""