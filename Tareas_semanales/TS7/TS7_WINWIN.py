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

# Banda de atenuación (fuera de lo últil)
f_stop = np.array([.1, 50.])  # Hz

# Banda pasante deseada (contenido últil del ECG)
f_pass = np.array([1.0, 30.0])  # Hz

# Ripple y atenuación para IIR (uso de filtfilt)
ripple_iir_db = 0.5  # dB
atten_iir_db = 20    # dB

# Ripple y atenuación para FIR (uso de lfilt)
ripple_fir_db = 1  # dB
atten_fir_db = 40  # dB

#%% PLANTILLA DE DISEÑO

if GRAFICAR:
    plt.figure(figsize=(10, 4))
    
    # Creamos un gráfico vacío donde se dibujará la plantilla la función se encarga de superponer 
    # zonas de paso y rechazo para un filtro determinado, como acá solo queremos la plantilla
    # vacía es necesario establecer manualmente los límites para que plt.axis() funcione
    # bien dentro de plot_plantilla
    plt.axis([0, fs / 2, -atten_fir_db - 10, ripple_fir_db + 5])
    
    plot_plantilla(filter_type='bandpass',
               fpass=f_pass,
               ripple=ripple_fir_db,
               fstop=f_stop,
               attenuation=atten_fir_db,
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
mi_iir_1 = sig.iirdesign(wp=f_pass, ws=f_stop, gpass=ripple_iir_db, gstop=atten_iir_db,
                         analog=False, ftype='cheby2', output='sos', fs=fs)
mi_iir_2 = sig.iirdesign(wp=f_pass, ws=f_stop, gpass=ripple_iir_db, gstop=atten_iir_db,
                         analog=False, ftype='ellip', output='sos', fs=fs)

# Grilla para barrido en frecuencia
w, hh = sig.sosfreqz(mi_iir_1, worN=2048, fs=fs)

# Gráfico módulo y fase

if GRAFICAR:
    graficar_filtro_sos(mi_iir_1, fs, f_pass, f_stop, ripple_iir_db, atten_iir_db, etiqueta='IIR - Cheby2')
    graficar_filtro_sos(mi_iir_2, fs, f_pass, f_stop, ripple_iir_db, atten_iir_db, etiqueta='IIR - Cauer')

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

#%% DISEÑO FIR - FIRWIN2 y FIRWIN1

cant_coef_firwin2 = 22501
cant_coef_firwin = 55501

freq = [0, f_stop[0], f_pass[0], f_pass[1], f_stop[1], fs/2]
gain = [0, 0, 1, 1, 0, 0]

mi_fir_1 = sig.firwin2(numtaps=cant_coef_firwin2, freq=freq, gain=gain, fs=fs,
                       window='hamming')
mi_fir_2 = sig.firwin(numtaps=cant_coef_firwin, cutoff=f_pass, window='hamming',
                      pass_zero=False, fs=fs)


if GRAFICAR:
    graficar_fir(mi_fir_1, fs, f_pass, f_stop, ripple_fir_db, atten_fir_db, etiqueta='FIR - firwin2 (Hamming)')
graficar_fir(mi_fir_2, fs, f_pass, f_stop, ripple_fir_db, atten_fir_db, etiqueta='FIR - firwin (Blackman)')

#%% APLICACIÓN DE LOS FILTROS Y COMPARACIÓN VISUAL

ecg_iir_cheby2 = sig.sosfiltfilt(mi_iir_1, ecg_one_lead)
ecg_iir_ellip = sig.sosfiltfilt(mi_iir_2, ecg_one_lead)

ecg_fir_firwin2 = sig.lfilter(mi_fir_1, [1], ecg_one_lead)
ecg_fir_firwin = sig.lfilter(mi_fir_2, [1], ecg_one_lead)
# Guardar las señales filtradas individualmente
np.save("ecg_iir_cheby2.npy", ecg_iir_cheby2)
t_ecg = np.arange(N_ecg) / fs_ecg
t_ini, t_fin = 0, N_ecg / fs_ecg
idx_ini, idx_fin = int(t_ini * fs_ecg), int(t_fin * fs_ecg)

plt.figure(figsize=(12, 6))
plt.plot(t_ecg[idx_ini:idx_fin], ecg_one_lead[idx_ini:idx_fin], label="ECG crudo",
         color='gray', linewidth=1)
plt.plot(t_ecg[idx_ini:idx_fin], ecg_iir_cheby2[idx_ini:idx_fin], label="IIR - Cheby2",
         linewidth=1, alpha=0.5)
# plt.plot(t_ecg[idx_ini:idx_fin], ecg_iir_ellip[idx_ini:idx_fin], label="IIR - Cauer",
#          linewidth=1, alpha=0.5)
# plt.plot(t_ecg[idx_ini:idx_fin], ecg_fir_firwin2[idx_ini:idx_fin], label="FIR - firwin2",
#          linewidth=1, alpha=0.5)
# plt.plot(t_ecg[idx_ini:idx_fin], ecg_fir_firwin[idx_ini:idx_fin], label="FIR - firwin", 
#          color='red', linewidth=1, alpha=0.5)
plt.title("Comparación de la señal ECG filtrada con distintos métodos")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%###############################
## Regiones de interés con ruido #
##################################
 
regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N_ecg, ii[1]]), dtype='uint')
   
    # plt.figure(1)
    # plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    # plt.plot(zoom_region, ecg_iir_cheby2[zoom_region], label='IIR - cheby2')
    # #plt.plot(zoom_region, ecg_iir_ellip[zoom_region + demora], label='IIR - ellip')
    # #plt.plot(zoom_region, ecg_fir_firwin2[zoom_region + demora], label='FIR - firwin2 (haming)')
    # #plt.plot(zoom_region, ecg_fir_firwin[zoom_region + demora], label='FIR - firwin (blackman)')
   
    # plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    # plt.ylabel('Adimensional')
    # plt.xlabel('Muestras (#)')
   
    # axes_hdl = plt.gca()
    # axes_hdl.legend()
    # axes_hdl.set_yticks(())
           
    #plt.show()