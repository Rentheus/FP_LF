#! /usr/bin/env python3
# -*- coding: utf-8 -*-

### FP Physik
#
# Tobias Sommer, 445306
# Axel Andrée, 422821

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%% Daten Laden, struktur: R_pt, R_C, R_Cu, R_Ta, R_Si, Tpt, Tc

data_raum = np.genfromtxt("LF_Raum.txt")
data_77K = np.genfromtxt("LF_77K.txt")
data_N = np.genfromtxt("LF_N.txt")
data_He = np.genfromtxt("LF_He.txt")[1:,:]
data_Noise = np.genfromtxt("LF_Noise.txt")


#%% kalibration

def f_T_pt(r, c1, c2):
    return c1*r + c2

def f_T_c(r, c1, c2, c3):
    return c1*np.exp(c2/r)+c3

data_r_pt = [np.mean(data_raum[:,0]),np.mean(data_77K[:,0])]
sigma_r_pt = [np.std(data_raum[:,0],ddof=1),np.std(data_77K[:,0],ddof=1)]
data_T_pt = [273.15+20.5, 77.15]

print(data_He[-10:,1])

data_r_c = [np.mean(data_raum[:,1]),np.mean(data_77K[:,1]), np.mean(data_He[-10:,1])]
sigma_r_c = [np.std(data_raum[:,1],ddof=1),np.std(data_77K[:,1],ddof=1), np.std(data_He[-10:,1],ddof=1)]
data_T_c = [273.15+20.5, 77.15, 4.15]


fig, ax = fig, ax = plt.subplots(1, 2, figsize=(10,4), layout = "tight")
ax[0].hist(data_raum[:,0], 6)
ax[0].axvline(data_r_pt[0], color = "r", label = "Mean")
ax[0].axvline(data_r_pt[0]-sigma_r_pt[0], color = "orange", label = "Standard Deviation", ls = ":")
ax[0].axvline(data_r_pt[0]+sigma_r_pt[0], color = "orange", label = "Standard Deviation", ls = ":")
ax[0].scatter(data_raum[:,0], np.zeros_like(data_raum[:,0])+ 0.4, color = "darkblue", marker = "x", label = "Measurements" )
ax[0].legend()
ax[0].set_title("T = 293.65K, PT100")
ax[0].set_xlabel("R/Ohm")



ax[1].hist(data_77K[:,0], 6)
ax[1].axvline(data_r_pt[1], color = "r", label = "Mean")
ax[1].axvline(data_r_pt[1]-sigma_r_pt[1], color = "orange", label = "Standard Deviation", ls = ":")
ax[1].axvline(data_r_pt[1]+sigma_r_pt[1], color = "orange", label = "Standard Deviation", ls = ":")
ax[1].scatter(data_77K[:,0], np.zeros_like(data_77K[:,0])+ 0.4, color = "darkblue", marker = "x", label = "Measurements" )
ax[1].legend()
ax[1].set_title("T = 77.15K, PT100")
ax[1].set_xlabel("R/Ohm")
#plt.show()
plt.savefig("Hist_PT_100.svg")


fig, ax = fig, ax = plt.subplots(1, 3, figsize=(10,4), layout = "tight")

ax[0].hist(data_raum[:,1], 8)
ax[0].axvline(data_r_c[0], color = "r", label = "Mean")
ax[0].axvline(data_r_c[0]-sigma_r_c[0], color = "orange", label = "Standard Deviation", ls = ":")
ax[0].axvline(data_r_c[0]+sigma_r_c[0], color = "orange", label = "Standard Deviation", ls = ":")
ax[0].scatter(data_raum[:,1], np.zeros_like(data_raum[:,1])+ 0.4, color = "darkblue", marker = "x", label = "Measurements" )
ax[0].legend()
ax[0].set_title("T = 293.65K, C-Resistor")
ax[0].set_xlabel("R/Ohm")



ax[1].hist(data_77K[:,1], 8)
ax[1].axvline(data_r_c[1], color = "r", label = "Mean")
ax[1].axvline(data_r_c[1]-sigma_r_c[1], color = "orange", label = "Standard Deviation", ls = ":")
ax[1].axvline(data_r_c[1]+sigma_r_c[1], color = "orange", label = "Standard Deviation", ls = ":")
ax[1].scatter(data_77K[:,1], np.zeros_like(data_77K[:,1])+ 0.4, color = "darkblue", marker = "x", label = "Measurements" )
ax[1].legend()
ax[1].set_title("T = 77.15K, C-Resistor")
ax[1].set_xlabel("R/Ohm")



ax[2].hist(data_He[-10:,1], 8)
ax[2].axvline(data_r_c[2], color = "r", label = "Mean")
ax[2].axvline(data_r_c[2]-sigma_r_c[2], color = "orange", label = "Standard Deviation", ls = ":")
ax[2].axvline(data_r_c[2]+sigma_r_c[2], color = "orange", label = "Standard Deviation", ls = ":")
ax[2].scatter(data_He[-10:,1], np.zeros_like(data_He[-10:,1])+ 0.4, color = "darkblue", marker = "x", label = "Measurements" )
ax[2].legend()
ax[2].set_title("T = 4.15K, C-Resistor")
ax[2].set_xlabel("R/Ohm")

plt.savefig("Hist_C.svg")

#plt.show()

pt = curve_fit(f_T_pt, data_r_pt, data_T_pt)
c = curve_fit(f_T_c, data_r_c, data_T_c)

#print(pt)
#print(c)

T_pt = lambda r : f_T_pt(r, pt[0][0], pt[0][1])
T_c = lambda r : f_T_c(r, c[0][0], c[0][1], c[0][2])

data_t_pt_N = T_pt(data_N[:,0])
data_t_c_N = T_c(data_N[:,1])

data_t_pt_He = T_pt(data_He[:,0])
data_t_c_He = T_c(data_He[:,1])

plt.plot(data_t_pt_N, data_t_c_N)
plt.plot(data_t_pt_He, data_t_c_He)

#plt.plot(data_T_pt, data_T_pt)
plt.show()
plt.plot(data_t_pt_N,data_N[:,3])
plt.show()
plt.plot(data_t_c_N,data_N[:,3])
plt.show()
plt.plot(data_t_pt_N,data_N[:,0])
plt.show()
