#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 10:32:19 2025

@author: lmaru
"""

import numpy as np
import matplotlib.pyplot as plt

# Grilla de frecuencias normalizadas en [0, π]
omega = np.linspace(0, np.pi, 1024)

# Diccionario de coeficientes h[n] para cada inciso
sistemas = {
    'a': [1, 1, 1, 1],
    'b': [1, 1, 1, 1, 1],
    'c': [1, -1],
    'd': [1, 0, -1]
}

# Crear figuras
plt.figure(figsize=(12, 8))

# Armar la respuesta en frecuencia evaluando la suma: H(e^{jω}) = sum h[n] e^{-jωn}
for i, (clave, h) in enumerate(sistemas.items()):
    #Crea un array del mismo tamaño que omega para guardar la respuesta en frecuencia.
    H = np.zeros_like(omega, dtype=complex) 
    for n, hn in enumerate(h):
        H += hn * np.exp(-1j * omega * n)
    
    # Módulo y fase
    modulo = np.abs(H)
    fase = np.unwrap(np.angle(H))

    # Gráficos
    plt.subplot(4, 2, 2*i+1)
    plt.plot(omega, modulo)
    plt.title(f'Módulo $|T(e^{{jω}})|$ - Inciso {clave}')
    plt.xlabel('ω [rad/muestra]')
    plt.grid()

    plt.subplot(4, 2, 2*i+2)
    plt.plot(omega, fase)
    plt.title(f'Fase $∠T(e^{{jω}})$ - Inciso {clave}')
    plt.xlabel('ω [rad/muestra]')
    plt.grid()

plt.tight_layout()
plt.show()
