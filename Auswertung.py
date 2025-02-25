#! /usr/bin/env python3
# -*- coding: utf-8 -*-

### FP Physik
#
# Tobias Sommer, 445306
# Axel Andrée, 422821

import numpy as np
import matplotlib.pyplot as plt

#%% Daten Laden, struktur: R_pt, R_C, R_Cu, R_Ta, R_Si, Tpt, Tc

data_raum = np.genfromtxt("LF_Raum.txt")
data_77K = np.genfromtxt("LF_77K.txt")
data_N = np.genfromtxt("LF_N.txt")
data_He = np.genfromtxt("LF_He.txt")
data_Noise = np.genfromtxt("LF_Noise.txt")


#%% kalibration

def T_pt(r, c1, c2):
    return c1*r + c2

def T_c(r, c1, c2, c3):
    return c1*np.exp(c2/r)+c3

data_r_pt = [np.mean(data_raum[:,0]),np.mean(data_77K[:,0])]
sigma_r_pt = [np.std(data_raum[:,0],ddof=1),np.std(data_77K[:,0],ddof=1)]

data_r_c = [np.mean(data_raum[:,1]),np.mean(data_77K[:,1]), np.mean(data_He[-10:,1])]
sigma_r_c = [np.std(data_raum[:,1],ddof=1),np.std(data_77K[:,1],ddof=1), np.std(data_He[-10:,1],ddof=1)]

fig, ax = fig, ax = plt.subplots(1, 2, figsize=(10,4), layout = "tight")
ax[0].hist(data_raum[:,0], 6)
ax[0].axvline(data_r_pt[0], color = "r", label = "Mean")
ax[0].axvline(data_r_pt[0]-sigma_r_pt[0], color = "orange", label = "Standard Deviation", ls = ":")
ax[0].axvline(data_r_pt[0]+sigma_r_pt[0], color = "orange", label = "Standard Deviation", ls = ":")
ax[0].scatter(data_raum[:,0], np.zeros_like(data_raum[:,0])+ 0.4, color = "darkblue", marker = "x", label = "Measurements" )

ax[1].hist(data_77K[:,0], 6)
ax[1].axvline(data_r_pt[1], color = "r", label = "Mean")
ax[1].axvline(data_r_pt[1]-sigma_r_pt[1], color = "orange", label = "Standard Deviation", ls = ":")
ax[1].axvline(data_r_pt[1]+sigma_r_pt[1], color = "orange", label = "Standard Deviation", ls = ":")
ax[1].scatter(data_77K[:,0], np.zeros_like(data_77K[:,0])+ 0.4, color = "darkblue", marker = "x", label = "Measurements" )
plt.show()


fig, ax = fig, ax = plt.subplots(1, 3, figsize=(10,4), layout = "tight")

ax[0].hist(data_raum[:,1], 8)
ax[0].axvline(data_r_c[0], color = "r", label = "Mean")
ax[0].axvline(data_r_c[0]-sigma_r_c[0], color = "orange", label = "Standard Deviation", ls = ":")
ax[0].axvline(data_r_c[0]+sigma_r_c[0], color = "orange", label = "Standard Deviation", ls = ":")
ax[0].scatter(data_raum[:,1], np.zeros_like(data_raum[:,1])+ 0.4, color = "darkblue", marker = "x", label = "Measurements" )

ax[1].hist(data_77K[:,1], 8)
ax[1].axvline(data_r_c[1], color = "r", label = "Mean")
ax[1].axvline(data_r_c[1]-sigma_r_c[1], color = "orange", label = "Standard Deviation", ls = ":")
ax[1].axvline(data_r_c[1]+sigma_r_c[1], color = "orange", label = "Standard Deviation", ls = ":")
ax[1].scatter(data_77K[:,1], np.zeros_like(data_77K[:,1])+ 0.4, color = "darkblue", marker = "x", label = "Measurements" )

ax[2].hist(data_He[-10:,1], 8)
ax[2].axvline(data_r_c[2], color = "r", label = "Mean")
ax[2].axvline(data_r_c[2]-sigma_r_c[2], color = "orange", label = "Standard Deviation", ls = ":")
ax[2].axvline(data_r_c[2]+sigma_r_c[2], color = "orange", label = "Standard Deviation", ls = ":")
ax[2].scatter(data_He[-10:,1], np.zeros_like(data_He[-10:,1])+ 0.4, color = "darkblue", marker = "x", label = "Measurements" )



plt.show()