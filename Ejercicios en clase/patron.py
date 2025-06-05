#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 19:38:20 2025

@author: lmaru
"""

#%%IMPORTACIONES
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio
#%% LECTURA DE DATOS Y FILTRADO DEL ECG
fs_ecg = 1000 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo
mat_struct = sio.loadmat('./ECG_TP4.mat') #Leo el archivo
fpass = np.array([1.0, 35.0])
