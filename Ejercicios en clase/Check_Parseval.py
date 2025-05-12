#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 14:37:50 2025

@author: lmaru
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 13:54:16 2025

@author: lmaru
"""
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio

##Lectura##
fs_ecg = 1000 # Hz, la frecuencia de muestreo a la que fue sampleada la señal del archivo
mat_struct = sio.loadmat('./ECG_TP4.mat') #Leo el archivo
ecg_one_lead = mat_struct['ecg_lead'].flatten() #Tomo la parte que quiero
ecg_one_lead = ecg_one_lead[:12000] #Nos quedamos con una parte del espectro limpia pero significativa
x = ecg_one_lead.astype(np.float64) #fuerzo que el tipo de dato sea tipo float no hacer trae errores númericos

##Calculo del miembro izquierdo de Parseval##
energia_tiempo = np.sum(np.abs(x)**2)

##Calculo del miembro derecho de Parseval##
N = len(x) #Para aplicarl la escala que indica el teorema, la fft no lo hace por default
X = np.fft.fft(x) #Calculo fft
energia_freq = (1/N) * np.sum(np.abs(X)**2)

##Resultados##
Parseval = np.isclose(energia_tiempo, energia_freq)  # True
print(Parseval)#True

#x = ecg_one_lead  # señal discreta
#x = x / np.std(x)