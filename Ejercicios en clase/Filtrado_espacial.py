#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 21:51:54 2025

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
fs = 1000 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo
mat_struct = sio.loadmat('./ECG_TP4.mat') #Leo el archivo
ecg_one_lead = mat_struct['ecg_lead'].flatten() #Tomo la parte que quiero
N_ecg = len(ecg_one_lead)
latidos = mat_struct['qrs_detections'].flatten()
#%% Armado de las realizaciones

t_pre = 0.1   # segundos antes del latido
t_post = 0.3  # segundos después del latido

# Convertís 0.1 y 0.3 segundos a muestras
muestras_pre = int(t_pre * fs)
muestras_post = int(t_post * fs)
L = muestras_pre + muestras_post

latidos_segmentados = []

for idx in latidos:
    ini = idx - muestras_pre
    fin = idx + muestras_post

    # Evitamos recortes fuera de los límites de la señal
    if ini >= 0 and fin <= len(ecg_one_lead):
        segmento = ecg_one_lead[ini:fin]
        latidos_segmentados.append(segmento)

# Convertimos a array 2D
latidos_segmentados = np.array(latidos_segmentados)

aux=latidos_segmentados[100]
#%% Promediado y visualizacion

latidos_norm = latidos_segmentados / np.max(np.abs(latidos_segmentados), axis=1, keepdims=True)

latido_promedio = np.mean(latidos_norm, axis=0) # Latido promedio

t = np.arange(len(latido_promedio)) / fs# Eje temporal

plt.figure(figsize=(10, 4))

for i in range(1903):
    plt.plot(t, latidos_norm[i], alpha=0.4)

# Latido promedio
plt.plot(t, latido_promedio, color='red', linewidth=2, label='Latido promedio')

plt.title("Latidos alineados y su promedio")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

"""Separar los latidos normales de los ventriculares es fundamental porque al promediarlos 
juntos, se pierde la identidad morfológica de cada tipo. El promedio resultante mezcla dos 
patrones fisiológicamente distintos, generando una forma artificial que no representa ni a los 
latidos normales ni a los ventriculares. Por eso, separamos antes de promediar para preservar 
la interpretabilidad clínica."""

#%%Clasificación entre normales y ventriculares
# latidos_norm ya tiene cada latido normalizado a máx = 1
picos = np.max(latidos_norm, axis=1)  # vector de picos por latido

# Clasificación binaria según umbral
mask_normales = picos > 0.55
mask_ventriculares = picos <= 0.55

# Separación
latidos_normales = latidos_norm[mask_normales]
latidos_ventriculares = latidos_norm[mask_ventriculares]

# Promedios por clase
latido_prom_normal = np.mean(latidos_normales, axis=0)
latido_prom_ventricular = np.mean(latidos_ventriculares, axis=0)

plt.figure(figsize=(10, 4))
t = np.linspace(-t_pre, t_post, latido_prom_normal.size)

# Latidos individuales normales (en verde claro)
for latido in latidos_normales:
    plt.plot(t, latido, color='lightgreen', alpha=0.4)

# Latidos individuales ventriculares (en violeta claro)
for latido in latidos_ventriculares:
    plt.plot(t, latido, color='violet', alpha=0.3)

# Promedio de latidos normales (línea verde gruesa)
plt.plot(t, latido_prom_normal, color='green', linewidth=3, label='Promedio normal')

# Promedio de latidos ventriculares (línea violeta gruesa)
plt.plot(t, latido_prom_ventricular, color='purple', linewidth=3, label='Promedio ventricular')

plt.title("Promedios por clase de latido")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

