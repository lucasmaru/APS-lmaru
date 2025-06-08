#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 10:50:04 2025

@author: lmaru
"""
#%%IMPORTACIONES
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio

#%% CONFIGURACIÓN GLOBAL
GRAFICAR = False  # Cambiar a False si no se desea mostrar gráficos
plt.close('all')
#%%ECG
fs_ecg = 1000 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo
mat_struct = sio.loadmat('./ECG_TP4.mat') #Leo el archivo
ecg_one_lead = mat_struct['ecg_lead'].flatten() #Tomo la parte que quiero
N_ecg = len(ecg_one_lead)

#%% VISUALIZACIÓN 1
if GRAFICAR:
    # Eje temporal
    t_ecg = np.arange(N_ecg) / fs_ecg

    # Dominio temporal
    plt.figure(figsize=(12, 4))
    plt.plot(t_ecg, ecg_one_lead)
    plt.title("ECG crudo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Espectro con Welch
    f_ecg, PSD_ecg = sig.welch(ecg_one_lead, fs=fs_ecg, window='hamming',
                               nperseg=2048, noverlap=1024, detrend='constant')

    # Escala dB
    PSD_ecg_dB = 10 * np.log10(PSD_ecg)
    plt.figure(figsize=(8, 4)) 
    plt.plot(f_ecg, PSD_ecg_dB)
    plt.title("PDS del ECG método de Welch [dB]")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD [dB]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%% PARÁMETROS - FILTRADO

# Frecuencia de muestreo
fs = 1000  # Hz
nyq = fs / 2  # Frecuencia de Nyquist

# Banda de atenuación (fuera de lo útil)
f_stop = np.array([.1, 50.])  # Hz

# Banda pasante deseada (contenido útil del ECG)
f_pass = np.array([1.0, 30.0])  # Hz

# Ripple permitido en la banda pasante
ripple_db = 1  # dB

# Atenuación mínima en la banda de stop
atten_db = 40  # dB

# Normalizar frecuencias para diseño (frecuencias normalizadas entre 0 y 1)
wp = np.array(f_pass) / nyq
wa = np.array(f_stop) / nyq

#%% PLANTILLA DE DISEÑO

if GRAFICAR:
    plt.figure(figsize=(10, 4))
    
    # Creamos un gráfico vacío donde se dibujará la plantilla la función se encarga de superponer 
    # zonas de paso y rechazo para un filtro determinado, como acá solo queremos la plantilla
    # vacía es necesario establecer manualmente los límites para que plt.axis() funcione
    # bien dentro de plot_plantilla
    plt.axis([0, fs / 2, -atten_db - 10, ripple_db + 5])
    
    plot_plantilla(filter_type='bandpass',
               fpass=f_pass,
               ripple=ripple_db,
               fstop=f_stop,
               attenuation=atten_db,
               fs=fs)
    
    plt.title("Plantilla de diseño para el filtrado del ECG")
    plt.tight_layout()
    plt.show()

#%% FUNCIÓN: GRAFICAR RESPUESTA DE UN FILTRO SOS

def graficar_filtro_sos(sos, fs, f_pass, f_stop, ripple_db, atten_db, etiqueta='Filtro'):
    w, hh = sig.sosfreqz(sos, worN=2048, fs=fs)
    orden = 2 * sos.shape[0]
    hh_dB = 20 * np.log10(np.maximum(np.abs(hh), 1e-10))
    fase = np.unwrap(np.angle(hh))
    demora = -np.diff(fase) / np.diff(w)
    w_med = (w[1:] + w[:-1]) / 2

    plt.figure(figsize=(10, 6))

    # Módulo
    plt.subplot(2, 1, 1)
    plt.plot(w, hh_dB, label=f'{etiqueta} (orden {orden})')
    plot_plantilla(filter_type='bandpass', fpass=f_pass, ripple=ripple_db,
                   fstop=f_stop, attenuation=atten_db, fs=fs)
    plt.ylabel("Ganancia [dB]")
    plt.xlabel("Frecuencia [Hz]")
    plt.title(f"Respuesta en módulo - {etiqueta}")
    plt.grid()
    plt.legend()

    # Fase y demora
    plt.subplot(2, 1, 2)
    plt.plot(w, fase, label='Fase')
    plt.plot(w_med, demora, label='Demora de grupo')
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Fase [rad] / Demora [s]")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()    
#%% DISEÑO IIR - CHEBY2 + CAUER

# Diseño del filtro en formato SOS
mi_iir_1 = sig.iirdesign(wp=f_pass, ws=f_stop, gpass=ripple_db, gstop=atten_db,
                         analog=False, ftype='cheby2', output='sos', fs=fs)
mi_iir_2 = sig.iirdesign(wp=f_pass, ws=f_stop, gpass=ripple_db, gstop=atten_db,
                         analog=False, ftype='ellip', output='sos', fs=fs)

# Grilla para barrido en frecuencia
w, hh = sig.sosfreqz(mi_iir_1, worN=2048, fs=fs)

# Gráfico módulo y fase

if GRAFICAR:
    graficar_filtro_sos(mi_iir_1, fs, f_pass, f_stop, ripple_db, atten_db, etiqueta='IIR - Cheby2')
    graficar_filtro_sos(mi_iir_2, fs, f_pass, f_stop, ripple_db, atten_db, etiqueta='IIR - Cauer')

#%% FUNCIÓN: GRAFICAR RESPUESTA FIR
def graficar_fir(fir_coefs, fs, f_pass, f_stop, ripple_db, atten_db, etiqueta='FIR'):
    w_fir, hh_fir = sig.freqz(fir_coefs, worN=8000, fs=fs)
    hh_fir_dB = 20 * np.log10(np.maximum(np.abs(hh_fir), 1e-10))
    fase_fir = np.unwrap(np.angle(hh_fir))
    demora_fir = -np.diff(fase_fir) / np.diff(w_fir)
    w_fir_med = (w_fir[1:] + w_fir[:-1]) / 2

    plt.figure(figsize=(10, 6))

    # Módulo
    plt.subplot(2, 1, 1)
    plt.plot(w_fir, hh_fir_dB, label=f'{etiqueta} (orden {len(fir_coefs) - 1})')
    plot_plantilla(filter_type='bandpass', fpass=tuple(f_pass),
                   ripple=ripple_db, fstop=tuple(f_stop),
                   attenuation=atten_db, fs=fs)
    plt.ylabel("Ganancia [dB]")
    plt.xlabel("Frecuencia [Hz]")
    plt.title(f"Respuesta en módulo - {etiqueta}")
    plt.grid()
    plt.legend()

    # Fase y demora
    plt.subplot(2, 1, 2)
    plt.plot(w_fir, fase_fir, label='Fase')
    plt.plot(w_fir_med, demora_fir, label='Demora de grupo')
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Fase [rad] / Demora [s]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% DISEÑO FIR - FIRWIN2 Y REMEZ

# Orden común para ambos (impar)
cant_coef = 25002  # Orden = cant_coef - 1
orden_fir = cant_coef - 1

# Grilla de diseño (Hz)
freq = [0, f_stop[0], f_pass[0], f_pass[1], f_stop[1], fs/2]
gain = [0, 0, 1, 1, 0, 0]

# FIR con firwin2 y ventana Hamming
mi_fir_1 = sig.firwin2(numtaps=cant_coef, freq=freq, gain=gain, fs=fs, window='hamming')

# FIR con Parks-McClellan (remez)
# Notar que remez espera freq y gain normalizados a [0, 1]
bands_norm = [0, f_stop[0], f_pass[0], f_pass[1], f_stop[1], fs/2]
bands_norm = np.array(bands_norm) / (fs/2)
desired = [0, 1, 0]
mi_fir_2 = sig.remez(numtaps=cant_coef, bands=bands_norm, desired=desired)



#if GRAFICAR:
graficar_fir(mi_fir_1, fs, f_pass, f_stop, ripple_db, atten_db, etiqueta='FIR - firwin2 (Hamming)')
graficar_fir(mi_fir_2, fs, f_pass, f_stop, ripple_db, atten_db, etiqueta='FIR - remez')
