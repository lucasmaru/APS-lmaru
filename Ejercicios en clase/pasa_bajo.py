#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 19:51:29 2025

@author: lmaru
"""

import numpy as np
import matplotlib.pyplot as plt

w0 = 1
Q_values = [10, 3, 1, 0.707, 0.5]
w = np.logspace(-2, 2, 1000)  # de 0.01 a 100 rad/s

plt.figure(figsize=(8,5))

for Q in Q_values:
    s = 1j * w
    H = w0**2 / (s**2 + (w0/Q)*s + w0**2)
    plt.plot(w, 20 * np.log10(np.abs(H)), label=f"Q={Q}")

plt.xscale('log')
plt.title('Pasa bajo de segundo orden para diferentes Q (ω₀=1)')
plt.xlabel('ω [rad/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()
