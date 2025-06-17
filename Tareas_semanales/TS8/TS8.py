#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 22:20:15 2025

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

#%%FILTRADO DE MEDIANA
# Ventanas en milisegundos
ventana_1_ms = 200
ventana_2_ms = 600

# Conversión de ventanas a número impar de muestras
ventana_1 = int(ventana_1_ms * fs / 1000)
ventana_2 = int(ventana_2_ms * fs / 1000)

if ventana_1 % 2 == 0:
    ventana_1 += 1
if ventana_2 % 2 == 0:
    ventana_2 += 1

# Primer paso: filtro de mediana sobre la señal original
m = sig.medfilt(ecg_one_lead, kernel_size=ventana_1)

# Segundo paso: filtro de mediana sobre el resultado anterior
b_hat = sig.medfilt(m, kernel_size=ventana_2)

# Señal filtrada: ECG sin el movimiento de línea de base
ecg_filtrada = ecg_one_lead - b_hat

#%% PLOTEO RESULTADOS
GRAFICA_1=False
if GRAFICA_1:
    plt.figure(figsize=(12, 4))
    plt.plot(ecg_one_lead, label='Señal original',alpha=0.5)
    plt.plot(b_hat, label='Estimación b̂[n] (mov. línea base)', linewidth=1.5)
    plt.plot(ecg_filtrada, label='ECG filtrado (x̂[n])', linewidth=1, alpha=0.5)
    plt.xlim(0, fs*10)  # mostrar primeros 10 segundos
    plt.title('Filtro de Mediana Anidado')
    plt.xlabel('Muestra [n]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

#%% INTERPOLACIÓN SPLINE CÚBICA

from scipy.interpolate import CubicSpline

# Definimos el desplazamiento n0 en milisegundos y lo convertimos a muestras
n0_ms = 100
n0 = int(n0_ms * fs / 1000)

# Generamos los mi = ni - n0, asegurándonos de no irnos fuera del rango
mi = latidos - n0
mi = mi[(mi >= 0) & (mi < N_ecg)]

# Creamos el conjunto de puntos S = {(mi, s[mi])}
si = ecg_one_lead[mi]

# Interpolamos con splines cúbicos
spline_interp = CubicSpline(mi, si)

# Evaluamos la interpolación en todos los puntos de la señal
b_hat_spline = spline_interp(np.arange(N_ecg))

# Señal filtrada
ecg_filtrada_spline = ecg_one_lead - b_hat_spline

#%% PLOTEO
GRAFICA_2=False
if GRAFICA_2:
    plt.figure(figsize=(12, 4))
    plt.plot(ecg_one_lead, label='Señal original')
    plt.plot(b_hat_spline, label='Estimación b̂[n] spline', linewidth=2)
    plt.plot(ecg_filtrada_spline, label='ECG filtrado spline', linewidth=1)
    plt.xlim(0, fs*10)
    plt.xlabel('Muestra [n]')
    plt.ylabel('Amplitud')
    plt.title('Interpolación del movimiento de línea de base mediante splines cúbicos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

#%% FILTRO ADAPTADO - TS8 Consigna 3

# Normalizamos la señal y el patrón
ecg_norm = ecg_one_lead / np.std(ecg_one_lead)
patron_qrs = mat_struct['qrs_pattern1'].flatten()
patron_qrs = patron_qrs / np.std(patron_qrs)

# Aplicamos correlación (matched filter)
y_matched = np.correlate(ecg_norm, patron_qrs, mode='same')
y_matched = y_matched / np.std(y_matched)

# Detectamos picos en la salida del filtro adaptado
umbral = 0.15  # relativo a std = 1
periodo_refractario = int(0.3 * fs)  # 300 ms
latidos_detectados, _ = sig.find_peaks(y_matched, height=umbral, distance=periodo_refractario)

#%% EVALUACIÓN DE DESEMPEÑO
tolerancia = int(0.05 * fs)  # ±50 ms
TP = 0
FP = 0
FN = 0
flag_detectado = np.zeros(len(latidos_detectados), dtype=bool)

for real in latidos:
    dentro = np.abs(latidos_detectados - real) <= tolerancia
    if np.any(dentro):
        TP += 1
        flag_detectado[np.where(dentro)[0][0]] = True
    else:
        FN += 1

FP = np.sum(~flag_detectado)

sensibilidad = TP / (TP + FN)
vpp = TP / (TP + FP)

print(f"TP = {TP}, FN = {FN}, FP = {FP}")
print(f"Sensibilidad = {sensibilidad:.3f}")
print(f"VPP = {vpp:.3f}")

#%% PLOTEO
t_ecg = np.arange(N_ecg) / fs

GRAFICA_3=True
if GRAFICA_3:
    plt.figure(figsize=(12, 4))
    plt.plot(t_ecg, y_matched, label='Salida del filtro adaptado', color='orange')
    plt.plot(t_ecg[latidos_detectados], y_matched[latidos_detectados], 'rx', label='Detecciones')
    plt.plot(t_ecg, ecg_norm, label='ECG normalizado', alpha=0.5)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.title("Filtro Adaptado sobre el ECG")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

