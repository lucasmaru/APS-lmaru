#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 19:04:40 2025

@author: lmaru
"""
from splane import bodePlot
from scipy.signal import TransferFunction
import numpy as np


# Definición de parámetros
Wo = 1
#Q = 5 # para visualizar los cambios con distintos Q
Q = np.sqrt(2)/2

# Definición de la función de transferencia H(s) = s^2 / (s^2 + Wo/Q s + Wo^2)
numerador = [1, 0, 0]  
denominador = [1, Wo/Q, Wo**2]

H = TransferFunction(numerador, denominador)

# Gráfica de Bode
bodePlot(H)
