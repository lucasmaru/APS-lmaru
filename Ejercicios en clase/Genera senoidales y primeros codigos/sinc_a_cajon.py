#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:21:46 2025

@author: lmaru
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinc en el tiempo con duración N·Ts y su espectro. El ancho espectral es 1/T por diseño.
@author: lmaru
"""

import numpy as np
import matplotlib.pyplot as plt

def grafica_sinc_y_espectro(fs, N, center=True, dB=False):
    ts = 1 / fs
    T = N * ts
    tt = np.linspace(-T/2, T/2, N).flatten()

    sinc_time = np.sinc(tt / ts) # sinc centrada

    plt.figure()

    # ---- Tiempo ----
    plt.subplot(2, 1, 1)
    plt.plot(tt, sinc_time)
    plt.title(f"Sinc en el tiempo - Duración total {T:.3f} s")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)

    # ---- Espectro ----
    M = 2048
    ft = 1/N * np.fft.fft(sinc_time, n=M)

    if center:
        ft = np.fft.fftshift(ft)
        ff = np.fft.fftshift(np.fft.fftfreq(M, d=ts))
    else:
        df = fs / M
        ff = np.linspace(0, (M - 1) * df, M)

    plt.subplot(2, 1, 2)
    if dB:
        plt.plot(ff, 20 * np.log10(np.abs(ft) / np.max(np.abs(ft)) + 1e-12))
        plt.ylabel("Magnitud (dB)")
    else:
        plt.plot(ff, np.abs(ft) / np.max(np.abs(ft)))
        plt.ylabel("Magnitud")

    if center:
        plt.title("Espectro centrado")
        if dB:
            plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='0 Hz')
    else:
        plt.title("Espectro no centrado")
        if dB:
            plt.axvline(x=fs/2, color='gray', linestyle='--', linewidth=1, label='fs/2')

    plt.xlabel("Frecuencia (Hz)")
    if dB:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Ejemplo de uso: cambiá N y fs y observá cómo cambia el ancho espectral
grafica_sinc_y_espectro(fs=50, N=100, center=False, dB=True)
#grafica_sinc_y_espectro(fs=1000, N=400, center=True, dB=True)
