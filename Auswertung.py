#! /usr/bin/env python3
# -*- coding: utf-8 -*-

### FP Physik
#
# Tobias Sommer, 445306
# Axel Andrée, 422821

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import iminuit

#%% Daten Laden, struktur: R_pt, R_C, R_Cu, R_Ta, R_Si, Tpt, Tc

data_raum = np.genfromtxt("LF_Raum.txt")
data_77K = np.genfromtxt("LF_77K.txt")
data_N = np.genfromtxt("LF_N.txt")
data_He = np.genfromtxt("LF_He.txt")[1:,:]
data_Noise = np.genfromtxt("LF_Noise.txt")


#%% kalibration

def f_R_pt(t, c1, c2):
    return(c1*t + c2)

def f_T_pt(r, c1, c2):
    return (r - c2)/c1

def f_R_c(t, c1, c2, c3):
    return c1*np.exp(c2/t)+c3

def f_T_c(r, c1, c2, c3):
    return (c2/(np.log((r-c3)/c1)))
    r#eturn (c2/(np.log(())))
    

    
data_r_pt = np.array([np.mean(data_raum[:,0]),np.mean(data_77K[:,0])])
sigma_r_pt = np.array([np.std(data_raum[:,0],ddof=1),-np.std(data_77K[:,0],ddof=1)])
data_T_pt = [273.15+20.5, 77.15]

#print(data_He[-10:,1])
#print(data_raum[:,1])
#print(data_77K[:,1])

data_r_c = np.array([np.mean(data_raum[:,1]),np.mean(data_77K[:,1]), np.mean(data_He[-10:,1])])
sigma_r_c = np.array([np.std(data_raum[:,1],ddof=1),np.std(data_77K[:,1],ddof=1), np.std(data_He[-10:,1],ddof=1)])
data_T_c = [273.15+20.5, 77.15, 4.15]
print(data_r_c)


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
plt.savefig("Hist_PT_100.pdf")


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

plt.savefig("Hist_C.pdf")

plt.show()
pt = curve_fit(f_R_pt, data_T_pt, data_r_pt)
c = curve_fit(f_R_c, data_T_c, data_r_c)

#print(pt)
#print(c)

T_pt = lambda r : f_T_pt(r, pt[0][0], pt[0][1])
T_c = lambda r : f_T_c(r, c[0][0], c[0][1], c[0][2])

print("Hier!")
print(T_c(3616))
print(T_pt(210))

data_t_pt_N = T_pt(data_N[:,0])
data_t_c_N = T_c(data_N[:,1])

data_t_pt_He = T_pt(data_He[:,0])
data_t_c_He = T_c(data_He[:,1])

plt.scatter(data_t_pt_N, data_t_c_N, label = "Measurements in N")
plt.scatter(data_t_pt_He, data_t_c_He, label= "Measurements in He")
plt.plot(data_t_pt_N, data_t_pt_N,ls = ":",  label = "expected Linearity in N")


plt.legend()
plt.ylabel("T_C / K")
plt.xlabel("T_Pt / K")
plt.title("Comparison T_Pt vs T_C")

plt.savefig("T_c_T_pt.pdf")
#plt.plot(data_T_pt, data_T_pt)
plt.show()



#%% Sys abweichungen
pt_minus = curve_fit(f_R_pt, data_T_pt , data_r_pt - sigma_r_pt)
c_minus = curve_fit(f_R_c, data_T_c, data_r_c - sigma_r_c)

pt_plus = curve_fit(f_R_pt, data_T_pt, data_r_pt + sigma_r_pt)
c_plus = curve_fit(f_R_c, data_T_c, data_r_c + sigma_r_c)


T_pt_minus = lambda r : f_T_pt(r, pt_minus[0][0], pt_minus[0][1])
T_c_minus = lambda r : f_T_c(r, c_minus[0][0], c_minus[0][1], c_minus[0][2])

data_t_pt_N_minus = T_pt_minus(data_N[:,0])
data_t_c_N_minus = T_c_minus(data_N[:,1])
data_t_c_He_minus = T_c_minus(data_He[:,1])

T_pt_plus = lambda r : f_T_pt(r, pt_plus[0][0], pt_plus[0][1])
T_c_plus = lambda r : f_T_c(r, c_plus[0][0], c_plus[0][1], c_plus[0][2])

data_t_pt_N_plus = T_pt_plus(data_N[:,0])
data_t_c_N_plus = T_c_plus(data_N[:,1])
data_t_c_He_plus = T_c_plus(data_He[:,1])

fig, ax = fig, ax = plt.subplots(2, 2, figsize=(9,7), layout = "tight")
fig.suptitle("Evaluation of Reference Point based Systematic Error in T")


ax[0,0].scatter(data_t_pt_N, data_t_pt_N_minus, label = "T_pt_minus")
ax[0,0].scatter(data_t_pt_N, data_t_pt_N_plus, label = "T_pt_plus")
ax[0,0].legend()
ax[0,0].set_ylabel("T / K")
ax[0,0].set_xlabel("T / K")


ax[1,0].scatter(data_t_pt_N, data_t_pt_N - data_t_pt_N_minus, label = "T_pt - T_pt_minus")
ax[1,0].scatter(data_t_pt_N, data_t_pt_N - data_t_pt_N_plus, label = "T_pt - T_pt_plus")
ax[1,0].legend()
ax[1,0].set_ylabel("delta T / K")
ax[1,0].set_xlabel("T / K")




ax[0,1].scatter(data_t_c_N, data_t_c_N_minus, label = "T_c_minus in N")
ax[0,1].scatter(data_t_c_N, data_t_c_N_plus, label = "T_c_plus in N")

ax[0,1].scatter(data_t_c_He, data_t_c_He_minus, label = "T_c_minus in He")
ax[0,1].scatter(data_t_c_He, data_t_c_He_plus, label = "T_c_plus in He")
ax[0,1].legend()
ax[0,1].set_ylabel("T / K")
ax[0,1].set_xlabel("T / K")




ax[1,1].scatter(data_t_c_N, data_t_c_N - data_t_c_N_minus, label = "T_c - T_c_minus in N")
ax[1,1].scatter(data_t_c_N, data_t_c_N - data_t_c_N_plus, label = "T_c - T_c_plus in N")

ax[1,1].scatter(data_t_c_He, data_t_c_He - data_t_c_He_minus, label = "T_c - T_c_minus in He")
ax[1,1].scatter(data_t_c_He, data_t_c_He - data_t_c_He_plus, label = "T_c - T_c_plus in He")
ax[1,1].legend()
ax[1,1].set_ylabel("delta T / K")
ax[1,1].set_xlabel("T / K")


plt.savefig("Sys_err_T_durch_Messwerte.pdf")


plt.show()
#plt.scatter(data_t_pt_N_plus, data_t_c_N_minus, label = "plus")
#plt.scatter(data_t_pt_N, data_t_c_N, label = "Mean")
#plt.scatter(data_t_pt_N_minus, data_t_c_N_plus, label = "Minus")

#plt.legend()
#plt.ylabel("T_C / K")
#plt.xlabel("T_Pt / K")
#plt.title("Comparison T_Pt vs T_C, sys")

#plt.show()

#%% Approx r zum vergleich
print(c[0])
print(pt[0])
R_c = lambda r : f_R_c(r, c[0][0], c[0][1], c[0][2])

R_approx = R_c(T_pt(data_N[:,0]))
R_approxhe = R_c(T_c(data_He[:,1]))
data_t_c_he = T_c(data_He[:,1])

plt.plot(data_t_pt_N, R_approx)
plt.scatter(data_t_pt_N, data_N[:,1])

plt.plot(data_t_c_he, R_approxhe)
plt.scatter(data_t_c_he, data_He[:,1])
plt.show()

r_extended_c = data_N[:, 1] 
T_extended_c = T_pt(data_N[:, 0])

c_2 =  curve_fit(f_R_c, T_extended_c, r_extended_c)

T_c_2 = lambda r : f_T_c(r, c_2[0][0], c_2[0][1], c_2[0][2])

print(T_c_2(3616))

R_c_2 = lambda r : f_R_c(r, c_2[0][0], c_2[0][1], c_2[0][2])
R_approx_2 = R_c_2(T_pt(data_N[:,0]))
R_approx_4k = R_c_2(4.15)
plt.plot(data_t_pt_N, R_approx_2)
plt.scatter(data_t_pt_N, data_N[:,1])
plt.scatter(T_c_2(data_r_c[-1]),data_r_c[-1], color = "red")
plt.scatter(4.15, data_r_c[-1], color = "green")
print(T_c_2(data_r_c[-1]))


plt.show()

data_t_c_N = T_c_2(data_N[:,1])

data_t_c_He = T_c_2(data_He[:,1])

plt.scatter(data_t_pt_N, data_t_c_N, label = "Measurements in N")
plt.scatter(data_t_pt_He, data_t_c_He, label= "Measurements in He")
plt.plot(data_t_pt_N, data_t_pt_N,ls = ":",  label = "expected Linearity in N")


plt.legend()
plt.ylabel("T_C / K")
plt.xlabel("T_Pt / K")
plt.title("Comparison T_Pt vs T_C_2")

plt.savefig("T_c_T_pt2.pdf")
#plt.plot(data_T_pt, data_T_pt)
plt.show()                       
#plt.plot(data_t_pt_N,data_N[:,3])
#plt.show()
#plt.plot(data_t_c_N,data_N[:,3])
#plt.show()
#plt.plot(data_t_pt_N,data_N[:,0])
#plt.show()



def T(messwerte):
    Temperature = np.zeros(messwerte.shape[0])
    for i in range(len(Temperature)):
        Temperature[i] = {True:T_pt(messwerte[i,0]), False: T_c(messwerte[i,1])}[messwerte[i,0]>data_T_pt[1]]
        #print(messwerte[i,0]>22.5)

    return Temperature

def t_err(messwerte, pt_err, c_err):
    Terr = np.zeros(messwerte.shape[0])
    for i in range(len(Terr)):
        Terr[i]={
            True: abs(pt_err* 1/pt[0][0]),
            False: abs(c_err* c[0][1] /(np.log((messwerte[i,1]- c[0][2])/c[0][0])**2 * (messwerte[i,1]- c[0][2])) )
            }


T_N = T(data_N)
T_He = T(data_He)

Si_filter = data_He[:,4]<0.5*10**38
#print(Si_filter)

plt.scatter(1/(T_N), np.log(1/data_N[:,4]))
plt.scatter(1/(T_He[Si_filter]), np.log(1/(data_He[:,4][Si_filter])))

#plt.scatter(T_N, data_N[:,4])
#plt.scatter(T_He[Si_filter], data_He[:,4][Si_filter])
#plt.yscale("log")
#plt.xscale("log")
plt.show()

plt.scatter(T_He[-300:], data_He[-300:,2], label = "CU")
plt.scatter(T_He[-300:], data_He[-300:,3], label = "'TA' + CU")
plt.legend()
plt.ylabel("$R  [\Omega]$")
plt.xlabel("$T  [K]$")
plt.show()




