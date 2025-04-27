#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 12:17:28 2025

@author: lmaru
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

fs = 1000         # frecuencia de muestreo (Hz)
N = 1000         # cantidad de muestras

df = fs / N
ts = 1 / fs

tt = np.linspace(0, (N-1)*ts, N)   # vector de tiempo
ff = np.linspace(0, (N-1)*df, N)


fa = 1 # Hz

# Genero la señal
analog_sig = np.sin(2 * np.pi * fa * tt)
#analog_sig += np.random.normal(0.0 , 1.0, N)
analog_sig += np.random.uniform(-0.001, +0.001, N)
# Normalizo en potencia
#analog_sig = analog_sig / np.sqrt(np.var(analog_sig))

# Calculo autocorrelación
autocor = sig.correlate(analog_sig, analog_sig)
lags = sig.correlation_lags(N, N)

plt.figure(1)

plt.plot(tt, analog_sig, linestyle='', color='blue', marker='o', markersize=2, markerfacecolor='blue')

plt.title('Señal')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

plt.figure(2)

plt.plot(lags, autocor)

plt.title('Autocor')
plt.xlabel('Lag en muestras [#]')
plt.ylabel('Autocorrelación [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

plt.figure(3)

#DEP via fft
fft_as= np.fft.fft(analog_sig)
plt.plot(ff[:N//2], 20*np.log10(np.abs(fft_as[:N//2])), label='DEP via FFT')

#DEP via autocor
fft_autocor=np.fft.fft(autocor)
plt.plot(ff[:N//2],10*np.log10(np.abs(fft_autocor[:N//2])), label='DEP via autocorr')

plt.title('Espectro')
plt.xlabel('frec [Hz]')
plt.ylabel('Módulo espectro [V]')
plt.legend()
axes_hdl = plt.gca()
plt.show()
