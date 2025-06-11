#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 21:47:16 2025

@author: lmaru
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla

#%% PARÁMETROS
fs = 1000  # Hz
f_hp = 1.0  # Hz
f_lp = 35.0  # Hz
orden_hp = 10001
orden_lp = 2001

#%% GRILLAS DE FRECUENCIA PARA FIRWIN2

# Pasa altos: ganancia 0 hasta 0.1 Hz, luego 1 desde f_hp hasta fs/2
freq_hp = [0, 0.1, f_hp, fs/2]
gain_hp = [0, 0,   1,   1]

# Pasa bajos: ganancia 1 hasta f_lp, luego 0 desde 50 Hz en adelante
freq_lp = [0, f_lp, 50.0, fs/2]
gain_lp = [1,   1,     0,     0]

#%% DISEÑO FIR CON FIRWIN2

fir_hp = sig.firwin2(numtaps=orden_hp, freq=freq_hp, gain=gain_hp, fs=fs,
                     window='hamming')
fir_lp = sig.firwin2(numtaps=orden_lp, freq=freq_lp, gain=gain_lp, fs=fs,
                     window='hamming')

#%% RESPUESTA EN FRECUENCIA COMBINADA
npoints = 8192
w, H_hp = sig.freqz(fir_hp, worN=npoints, fs=fs)
_, H_lp = sig.freqz(fir_lp, worN=npoints, fs=fs)# El _ es el mismo w

H_total = H_hp * H_lp

#%% GRAFICAR PLANTILLA Y RESPUESTA
plt.figure(figsize=(10, 5))
plt.plot(w, 20 * np.log10(np.abs(H_total) + 1e-12), 
         label=f'FIR orden hp {orden_hp - 1} y orden lp {orden_lp - 1}')
plot_plantilla(filter_type='bandpass', fpass=np.array([f_hp, f_lp]), 
               ripple=1, attenuation=40, fstop=np.array([0.1, 50.]), fs=fs)
plt.title('Respuesta en frecuencia combinada con firwin2')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.legend()
plt.tight_layout()

#%% APLICACIÓN A SEÑAL ECG
mat_struct = sio.loadmat('./ECG_TP4.mat')
ECG = mat_struct['ecg_lead'].flatten()

# Cascada: primero pasa altos, luego pasa bajos
ECG_temp = sig.lfilter(fir_hp, 1, ECG)
ECG_filt = sig.lfilter(fir_lp, 1, ECG_temp)

#%% VISUALIZACIÓN ECG
# t = np.arange(len(ECG)) / fs

# plt.figure(figsize=(12, 6))
# plt.plot(t, ECG, label='ECG original', alpha=0.5)
# plt.plot(t, ECG_filt, label='ECG filtrada (firwin2 HP + LP)', color='orange')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud')
# plt.grid(True)
# plt.title('Filtrado del ECG en cascada con firwin2')
# plt.legend()
# plt.tight_layout()
