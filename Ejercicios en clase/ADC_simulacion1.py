#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:58:13 2025

@author: Lucas Marú
"""

#%% módulos y funciones a importar

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

#%% Datos de la simulación
fs=1000 # frecuencia de muestreo (Hz)
N =1000  # cantidad de muestras

# Datos del ADC
B = 8 # bits
Vf = 10# rango simétrico de +/- Vf Volts
q = (Vf)/2**(B-1) # paso de cuantización de q Volts

# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = (q**2)/12# Watts, intrinseco del ADC
kn = 10. # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn # Simulamos un ruido analogico

# %% Genera seno
ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral
f0 = 1# grilla de sampleo temporal 
tt = np.linspace(0, (N-1)*ts, N).flatten()
"""
   Busco los valores para los cuales esta senoidal esta normalizada, es decir 
   que la señal tenga una varianza = 1, recordar que relacionamos la vaianza 
   con la energia o potencia. 
   La senoidal es x(t) = A sen ( wt + phi ), la varianza va a estar dada por 
   Var(x) = A^2 * 1/2 por esto definimos A como la definimos 
"""
A = math.sqrt(2) # proceso de normalización 
xx = A*np.sin( 2 * np.pi * f0 * tt  ) # Declaro funcion senoidal
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(tt, xx)
var_xx = np.var(xx)
print(var_xx)

# %%Generador de ruido
# Distrubuido de forma Normal de media nula, le prescribo la potencia para que 
# sea la que definí en el primer bloque como pot_ruido_analog y la cantidad de 
# muestras tiene que ser la misma para que los vectores esten apareados en
# cantidad de elementos
sigma = math.sqrt(pot_ruido_analog) # Esta es la prescripción de la potencia
noise = np.random.normal(0,sigma,N) # media, sigma y cantidad
plt.subplot(3,1,2)
plt.plot(tt, noise)
var_n = np.var(noise) 
# imprimo para chequear que la potencia sea la que le prescribi
print(var_n)
print(pot_ruido_analog)

#%% Sumo la señal analógica con el ruido
xr = xx + noise
plt.subplot(3,1,3)
plt.plot(tt,xr)

#%% Cuantizo
"""
Dividimos la señal ruidosa xr por el paso de cuantización 𝑞: xr/q
Esto nos da una versión "escalada" de la señal, donde los valores ya no están 
en voltios, sino en unidades de 𝑞. Es decir, si 𝑞=0.078125 V y 𝑥𝑟= 0.32 V, 
obtenemos: 0.32 / 0.078125 ≈ 4.1
La función np.round() redondea cada valor al entero más cercano. Para el 
ejemplo anterior: round(4.1)=4
Esto significa que estamos asignando cada muestra de la señal al nivel de 
cuantización más cercano. Por último multiplicamos nuevamente por q para volver
a la escala original en voltios
"""
xq = q * np.round(xr / q) # esta sola linea modela la cuantizaciòn 

#%% Experimento: 
"""
   Se desea simular el efecto de la cuantización sobre una señal senoidal de 
   frecuencia 1 Hz. La señal "analógica" podría tener añadida una cantidad de 
   ruido gausiano e incorrelado.
   
   Se pide analizar el efecto del muestreo y cuantización sobre la señal 
   analógica. Para ello se proponen una serie de gráficas que tendrá que ayudar
   a construir para luego analizar los resultados.
   
"""


# Señales

analog_sig =xx # señal analógica sin ruido
sr = xr # señal analógica de entrada al ADC (con ruido analógico)
srq = xq # señal cuantizada

nn = xr - xx  # señal de ruido de analógico
nq = xq - xr  # señal de ruido de cuantización

#%% Visualización de resultados

# cierro ventanas anteriores
plt.close('all')
    
##################
# Señal temporal
##################

plt.figure(1)

plt.plot(tt, srq, lw=2, linestyle='', color='blue', marker='o', markersize=5, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='ADC out (diezmada)')
plt.plot(tt, sr, lw=1, color='black', marker='x', ls='dotted', label='$ s $ (analog)')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


###########
# Espectro
###########

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr)#escalamiento no es importante (1/N)
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2#vector booleano mitad True, mitad False

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

#############
# Histograma
#############

plt.figure(3)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')

