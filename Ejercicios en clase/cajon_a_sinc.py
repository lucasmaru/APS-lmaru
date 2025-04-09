#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización de la ventana rectangular (cajón) y su espectro, con centrado y escala dB opcionales.
@author: lmaru
"""
#%%Explicación
"""Cuando quiero graficar el espectro del cajon al estilo como lo veniamos haciendo en clases el espectro sale 
cortado, solo se visualiza media sinc.
Al intentar solucionar eso encontre un opción que es np.fft.fftshift que sirve justamente para reordenar el
vector de tal manera que grafique para valores que tienen como centro el cero, permitiendo que se vea el patrón
entero de la sinc. Cuando se shiftea ya no se usa el vector booleano bfreq porque el fin de este es justamente
tomar unicamente los valores positivos que devuelve la fft. Como detalle técnico cuando usamos shitf tambien hay
que reordenar la grilla de frecuencia con la misma lógica para que sea posible realizar el gráfico. La opción 
center da la opcioń de hacer el gráfico de la manera tradicional (center=False) o utilazdo shift en caso 
contrario.
Por último la opción dB permite graficar con el eje de frecuencias logaritmado o no, esta bueno para agarrar 
intuición de como se ve la sinc en dB
"""
#%% Importaciones
import numpy as np
import matplotlib.pyplot as plt
#%% Defino función
def grafica_cajon_y_espectro(fs, N, center=True, dB=False):
    ts = 1 / fs
    tt = np.linspace(0, (N - 1) * ts, N).flatten()
    
    window = np.ones(N)

    plt.figure()
    
    # ---- Tiempo ----
    plt.subplot(2,1,1)
    plt.stem(tt, window, basefmt=" ")# evita que dibuje una línea horizontal en y=0
    #Stem construye un gráfico que resalta la condición discreta de la señal
    plt.title(f"Cajón en el tiempo (ventana rectangular) - {N} muestras a {fs} Hz")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)

    # ---- Espectro ----
    M = 2048
    ft = 1/N * np.fft.fft(window, n=M)

    if center:
        ft = np.fft.fftshift(ft)
        ff = np.fft.fftshift(np.fft.fftfreq(M, d=ts))
    else:
        df = fs / M
        ff = np.linspace(0, (M - 1) * df, M)

    plt.subplot(2,1,2)

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

#%% Uso función
grafica_cajon_y_espectro(50, 100, center=True, dB=True)
grafica_cajon_y_espectro(51.5, 100, center=True, dB=True)
