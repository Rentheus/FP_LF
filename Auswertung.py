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
#print(data_r_c)


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

#print("Hier!")
#print(T_c(3616))
#print(T_pt(210))

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

#print(data_t_c_N - data_t_c_N_minus)


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
#print(c[0])
#print(pt[0])
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

#print(T_c_2(3616))

R_c_2 = lambda r : f_R_c(r, c_2[0][0], c_2[0][1], c_2[0][2])
R_approx_2 = R_c_2(T_pt(data_N[:,0]))
R_approx_4k = R_c_2(4.15)
plt.plot(data_t_pt_N, R_approx_2)
plt.scatter(data_t_pt_N, data_N[:,1])
plt.scatter(T_c_2(data_r_c[-1]),data_r_c[-1], color = "red")
plt.scatter(4.15, data_r_c[-1], color = "green")
#print(T_c_2(data_r_c[-1]))


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

        Temperature[i] = {True:T_pt(messwerte[i,0]), False: T_c(messwerte[i,1])}[messwerte[i,0]>data_r_pt[1]]

        #print(messwerte[i,0]>22.5)

    return Temperature

def t_err(messwerte, pt_err, c_err):
    Terr = np.zeros(messwerte.shape[0])
    for i in range(len(Terr)):
        Terr[i]={
            True: abs(pt_err[i]* 1/pt[0][0]),
            False: abs(c_err[i]* c[0][1] /(np.log((messwerte[i,1]- c[0][2])/c[0][0])**2 * (messwerte[i,1]- c[0][2])) )
            }[messwerte[i,0]>data_r_pt[1]]
    return Terr


T_N = T(data_N)
T_He = T(data_He)

#print(T_N)
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

#%%
def lfit(x, a, b):
    return a*x+b



Pt = data_N[:,0]
C = data_N[:,1]
Cu = data_N[:,2]
Ta = data_N[:,3]
Si = data_N[:,4]
#%%

l = 1
Intervalle_N = []

for x in range(0, len(data_N[:,0])):
    if x == len(data_N[:,0])-1:
        Intervalle_N.append(l)
    else:
        if data_N[:,0][x] - data_N[:,0][x+1] < 0.07:
            l += 1
        else:
            Intervalle_N.append(l)
            l = 1

Intervalle_N[len(Intervalle_N)-1] = 12
Intervalle_N.append(12)
#print(len(Intervalle_N))
#print(np.sum(Intervalle_N))

l = 1
Intervalle_He = []
Intervalle_He = [9, 12, 11, 10, 11, 10, 11, 10, 11, 11, 11, 11, 11, 11, 12, 11, 11, 11, 11, 13, 11, 12, 9, 8, 12, 14, 13, 11, 12, 10, 13, 11, 13, 12, 11, 12, 23, 18, 10, 11, 5, 10]
#for x in range(len(data_He[:,0])):
#    #print(x)
#    if x == len(data_He[:,0])-1:
#        Intervalle_He.append(l)
#    else:
#        print(abs(np.log(data_He[:,1][x])- np.log(data_He[:,1][x+1]))/data_He[:,1][x])
#        if abs(np.log(data_He[:,1][x])- np.log(data_He[:,1][x+1]))/data_He[:,1][x] < 0.0000027:
#            
#            l += 1
#        else:
#            Intervalle_He.append(l)
#            l = 1
#print(Intervalle_He)
#print(data_He[128:150, 1])
#print(data_He[138, 1])
#print(data_He[139, 1])
#print(abs(np.log(data_He[:,1][138])- np.log(data_He[:,1][139]))/data_He[:,1][138])


#print(data_He[-50:-60, 1])

#print(abs(np.log(data_He[:,1][-56])- np.log(data_He[:,1][-55]))/data_He[:,1][-56])


#plt.scatter(np.array(range(30)),  data_He[120:150,1])
#plt.show()

#plt.scatter(np.array(range(30)),  data_He[400:430,1])
#plt.show()
#plt.scatter(np.array(range(30)),  data_He[-30:,1])
#plt.show()
#temp  = 0

#plt.scatter(np.array(range(480)),  data_He[:,1], marker  = ".")
#temp = 0
#for i in range(len(Intervalle_He)):
#    print(np.array(range(Intervalle_He[i] +Intervalle_He[i+1])).shape)
#    print(data_He[temp:temp+Intervalle_He[i] +Intervalle_He[i+1], 1].shape)
#    plt.scatter(np.array(range(Intervalle_He[i] +Intervalle_He[i+1])), data_He[temp:temp+Intervalle_He[i] +Intervalle_He[i+1], 1 ])
#    plt.show()
#    temp += Intervalle_He[i]

#plt.title(str(i*30))



#print(sum(Intervalle_He))
#print(len(data_He[:,1]))
# Intervalle_He[len(Intervalle_He)-1] = 12
# Intervalle_He.append(12)
#print(len(Intervalle_He))
#print(np.sum(Intervalle_He))

#%%



def statsigma_N(Intervalle, Stoff, plot = "nein"):
    
    s=0
    
    statsigma_N = []
    
    
    if plot == "ja":
        for n in Intervalle:
            
            
            
            
            Pt_k = Pt[s:s+n]
            y_k = Stoff[s:s+n]
            
            params = curve_fit(lfit, T_pt(Pt_k), y_k)
            lfitn = lambda r : lfit(r, params[0][0], params[0][1])
            
            
            
            plt.scatter(T_pt(Pt_k), y_k)
            plt.plot(T_pt(Pt_k), lfitn(T_pt(Pt_k)))
            plt.show()
            
            
            
            zwischenergebnis = 0
            
            for i in range(0, len(Pt_k)):
                zwischenergebnis += ( y_k[i] - lfitn(T_pt(Pt_k[i])) )**2
            
            sig = np.sqrt( (1/(len(Pt_k) - 2)) * zwischenergebnis )
            
            
            
            s=s+n
            for i in range(n):
                statsigma_N.append(sig)
        return np.array(statsigma_N)
    
    else:
        for n in Intervalle:
            Pt_k = Pt[s:s+n]
            y_k = Stoff[s:s+n]
            
            params = curve_fit(lfit, T_pt(Pt_k), y_k)
            lfitn = lambda r : lfit(r, params[0][0], params[0][1])
            
            zwischenergebnis = 0
            
            for i in range(0, len(Pt_k)):
                zwischenergebnis += ( y_k[i] - lfitn(T_pt(Pt_k[i])) )**2
            
            sig = np.sqrt( (1/(len(Pt_k) - 2)) * zwischenergebnis )
            
            
            
            s=s+n
            for i in range(n):  
                statsigma_N.append(sig)
        return np.array(statsigma_N)
#%% statsigma he

def statsigma_He(Intervalle, Stoff, plot = "nein"):
    
    C_He = data_He[:,1]
    s=0
    
    statsigma_N = []
    
    
    if plot == "ja":
        for n in Intervalle:
            
            
            
            
            C_He_k = C_He[s:s+n]
            y_k = Stoff[s:s+n]
            
            params = curve_fit(lfit, T_c(C_He_k), y_k)
            lfitn = lambda r : lfit(r, params[0][0], params[0][1])
            
            
            
            plt.scatter(T_c(C_He_k), y_k)
            plt.plot(T_c(C_He_k), lfitn(T_c(C_He_k)))
            plt.show()
            
            
            
            zwischenergebnis = 0
            
            for i in range(0, len(C_He_k)):
                zwischenergebnis += ( y_k[i] - lfitn(T_c(C_He_k[i])) )**2
            
            sig = np.sqrt( (1/(len(C_He_k) - 2)) * zwischenergebnis )
            
            
            
            s=s+n
            for i in range(n):
                statsigma_N.append(sig)
        return np.array(statsigma_N)
    
    else:
        for n in Intervalle:
            C_He_k = C_He[s:s+n]
            y_k = Stoff[s:s+n]
            
            params = curve_fit(lfit, T_c(C_He_k), y_k)
            lfitn = lambda r : lfit(r, params[0][0], params[0][1])
            
            zwischenergebnis = 0
            
            for i in range(0, len(C_He_k)):
                zwischenergebnis += ( y_k[i] - lfitn(T_c(C_He_k[i])) )**2
            
            sig = np.sqrt( (1/(len(C_He_k) - 2)) * zwischenergebnis )
            
            
            
            s=s+n
            for i in range(n):  
                statsigma_N.append(sig)
        return np.array(statsigma_N)




#%%        
   
sig_Pt_N = statsigma_N(Intervalle_N, Pt)
sig_C_N = statsigma_N(Intervalle_N, C)
sig_Cu_N = statsigma_N(Intervalle_N, Cu)
sig_Ta_N = statsigma_N(Intervalle_N, Ta)
sig_Si_N = statsigma_N(Intervalle_N, Si)

sig_Pt_He = statsigma_He(Intervalle_He, data_He[:,0])
sig_C_He = statsigma_He(Intervalle_He, data_He[:,1], plot = "nein")
sig_Cu_He = statsigma_He(Intervalle_He, data_He[:,2], plot= "nein")
sig_Ta_He = statsigma_He(Intervalle_He, data_He[:,3], plot = "nein")
sig_Si_He = statsigma_He(Intervalle_He, data_He[:,4], plot= "ja")

#print(sig_Ta_He)

Terr_N = t_err(data_N, sig_Pt_N, sig_C_N)
Terr_He = t_err(data_He, sig_Pt_He, sig_C_He)

plt.scatter(T_N, Terr_N)
plt.scatter(T_He,  Terr_He)
plt.axvline(77.15)
plt.show()
print(sig_Pt_N[0])
print(Terr_N[0])


#print(sig_Si_He.shape)

#%% measured resistances vs T

####USE Si_filter for HE & Si

# all in one diagram
fig, ax = fig, ax = plt.subplots(5, 1, figsize=(10,15), layout = "tight", sharex = True)
ax[0].errorbar(T_N, data_N[:, 0], sig_Pt_N, Terr_N, label = "Resistances Pt100, N", fmt = ".")
ax[0].errorbar(T_He, data_He[:, 0], sig_Pt_He, Terr_He, label = "Resistances Pt100, He", fmt = ".")
ax[0].set_xlabel("$T [K]$")
ax[0].set_ylabel("$R [\Omega]$")
ax[0].title.set_text("Resistance Measurement Pt100")
ax[0].legend()

ax[1].errorbar(T_N, data_N[:, 1], sig_C_N, Terr_N, label = "Resistances C, N", fmt = ".")
ax[1].errorbar(T_He, data_He[:, 1], sig_C_He, Terr_He, label = "Resistances C, He", fmt = ".")
ax[1].set_xlabel("$T [K]$")
ax[1].set_ylabel("$R [\Omega]$")
ax[1].title.set_text("Resistance Measurement Carbon Resistor")
ax[1].legend()

ax[2].errorbar(T_N, data_N[:, 2], sig_Cu_N, Terr_N, label = "Resistances Cu, N", fmt = ".")
ax[2].errorbar(T_He, data_He[:, 2], sig_Cu_He, Terr_He, label = "Resistances Cu, He", fmt = ".")
ax[2].set_xlabel("$T [K]$")
ax[2].set_ylabel("$R [\Omega]$")
ax[2].title.set_text("Resistance Measurement Copper")
ax[2].legend()

ax[3].errorbar(T_N, data_N[:, 3], sig_Ta_N, Terr_N, label = "Resistances Ta, N", fmt = ".")
ax[3].errorbar(T_He, data_He[:, 3], sig_Ta_He, Terr_He, label = "Resistances Ta, He", fmt = ".")
ax[3].set_xlabel("$T [K]$")
ax[3].set_ylabel("$R [\Omega]$")
ax[3].title.set_text("Resistance Measurement Tantal, raw")
ax[3].legend()

ax[4].errorbar(T_N, data_N[:, 4], sig_Si_N, Terr_N, label = "Resistances Si, N", fmt = ".")
ax[4].errorbar(T_He[Si_filter], data_He[Si_filter][:, 4], sig_Si_He[Si_filter], Terr_He[Si_filter], label = "Resistances Si, He", fmt = ".")
ax[4].set_xlabel("$T [K]$")
ax[4].set_ylabel("$R [\Omega]$")
ax[4].title.set_text("Resistance Measurement Silicon")
ax[4].legend()

plt.savefig("measured_resistances.pdf")

plt.show()


#%% Tantalum problem


fig, ax = fig, ax = plt.subplots(2, 1, figsize=(9,7), layout = "tight")

ax[0].errorbar(T_N, data_N[:,3], sig_Ta_N, Terr_N,  label= "Measured Tantalum Resistances, N", fmt = ".", color = "deepskyblue")
ax[0].errorbar(T_He, data_He[:,3], sig_Ta_He, Terr_He, label= "Measured Tantalum Resistances, He" , fmt = ".", color = "blue")

ax[0].errorbar(T_N, data_N[:,2], sig_Cu_N, Terr_N,  label= "Measured Copper Resistances, N", fmt = ".", color = "orange")
ax[0].errorbar(T_He, data_He[:,2], sig_Cu_He, Terr_He, label= "Measured Copper Resistances, He" , fmt = ".", color = "darkorange")

ax[0].set_ylabel("$R [\Omega]$")
ax[0].set_xlabel("$T [K]$")
ax[0].title.set_text("Comparison Copper and Tantalum Resistance")
ax[0].legend()

Ta_small_filter = data_He[:,3]<1

Ta_small = data_He[:,3][data_He[:,3]<1]
Cu_small = data_He[:,2][data_He[:,3]<1]
T_Ta_small = T_He[Ta_small_filter]
ax[1].errorbar(T_Ta_small, Ta_small,np.array(sig_Ta_He)[Ta_small_filter], Terr_He[Ta_small_filter],  label= "Measured Tantalum Resistance", fmt = ".", color = "blue", alpha =0.8) 
ax[1].errorbar(T_Ta_small, Cu_small,np.array(sig_Cu_He)[Ta_small_filter], Terr_He[Ta_small_filter],  label= "Measured Copper Resistance", fmt = "x", color = "darkorange", alpha = 0.8)

ax[1].set_ylabel("$R [\Omega]$")
ax[1].set_xlabel("$T [K]$")
ax[1].legend()
plt.savefig("Ta_problem.pdf")

plt.show()

Ta_new_N = data_N[:,3] -data_N[:,2]
Ta_new_He = data_He[:,3] -data_He[:,2]


sig_Ta_new_N = np.sqrt(sig_Ta_N**2 + sig_Cu_N**2)
sig_Ta_new_He = np.sqrt(sig_Ta_He**2 + sig_Cu_He**2)

#%% Cu, Ta, diagrams


#Cu
fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,10), layout = "tight")

ax[0].errorbar(T_N, data_N[:, 2], sig_Cu_N, Terr_N, label = "Resistances Cu, N", fmt = ".")
ax[0].errorbar(T_He, data_He[:, 2], sig_Cu_He, Terr_He, label = "Resistances Cu, He", fmt = ".")
ax[0].set_xlabel("$T [K]$")
ax[0].set_ylabel("$R [\Omega]$")
ax[0].title.set_text("Resistance Measurement Cu")
ax[0].legend()

ax[1].errorbar(T_N, data_N[:, 2], sig_Cu_N, Terr_N, label = "Resistances Cu, N", fmt = ".")
ax[1].errorbar(T_He, data_He[:, 2], sig_Cu_He, Terr_He, label = "Resistances Cu, He", fmt = ".")
ax[1].set_xlabel("$T [K]$")
ax[1].set_ylabel("$R [\Omega]$")
#ax[1].title.set_text("Resistance Measurement Cu")
ax[1].set_yscale("log")
ax[1].set_xscale("log")

ax[1].legend()

plt.savefig("cu_lin_log_scale.pdf")
plt.show()


#Ta
fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,10), layout = "tight")

ax[0].errorbar(T_N, Ta_new_N, sig_Ta_new_N, Terr_N, label = "Resistances Ta, N", fmt = ".")
ax[0].errorbar(T_He, Ta_new_He, sig_Ta_new_He, Terr_He, label = "Resistances Ta, He", fmt = ".")
ax[0].set_xlabel("$T [K]$")
ax[0].set_ylabel("$R [\Omega]$")
ax[0].title.set_text("Resistance Measurement Ta")
ax[0].legend()

ax[1].errorbar(T_N, Ta_new_N, sig_Ta_new_N, Terr_N, label = "Resistances Ta, N", fmt = ".")
ax[1].errorbar(T_He, Ta_new_He, sig_Ta_new_He, Terr_He, label = "Resistances Ta, He", fmt = ".")
ax[1].set_xlabel("$T [K]$")
ax[1].set_ylabel("$R [\Omega]$")
#ax[1].title.set_text("Resistance Measurement Cu")
ax[1].set_yscale("log")
ax[1].set_xscale("log")

ax[1].legend()
plt.savefig("Ta_lin_log_scale.pdf")
plt.show()

#%% linear regime cu

# fit  R = R_0(1 -alpha*T ), T in C


#join data for he and N
T_joined = np.concatenate((T_N, T_He))
T_joined_err = np.concatenate((Terr_N, Terr_He))

Cu_joined = np.concatenate((data_N[:,2], data_He[:,2]))
Cu_joined_err = np.concatenate((sig_C_N, sig_C_He))

Cu_lin_selector = Cu_joined>6
    
#cu to T_raum 
T_joined = T_joined - 273.15

def lin_approx(T, R_0, alpha):
    return R_0*(1-alpha*T)

def chi2_lin(fit_func, x, y, xerr, yerr, R_0, alpha):
    'chi2 für linearen fit'
    chi2_value = 0
    #for i in range(len(x)):
    model = fit_func(x, R_0 = R_0, alpha = alpha)
    chi2_value = np.sum(((y - model) / np.sqrt(yerr**2 + (np.gradient(model, x) * xerr)**2))**2)
    return chi2_value



chi_lin_cu = lambda R_zero, Alpha: chi2_lin(lin_approx, T_joined[Cu_lin_selector], Cu_joined[Cu_lin_selector], T_joined_err[Cu_lin_selector], Cu_joined_err[Cu_lin_selector], R_zero, Alpha)


m_cu = iminuit.Minuit(chi_lin_cu, R_zero = 22.5, Alpha = -0.003)
print(m_cu.migrad())
print(m_cu.errors)
print(m_cu.fval)
model = lin_approx(T_joined[Cu_lin_selector], m_cu.values["R_zero"], m_cu.values["Alpha"])
#plt.errorbar(T_joined[Cu_lin_selector], Cu_joined[Cu_lin_selector],Cu_joined_err[Cu_lin_selector], T_joined_err[Cu_lin_selector], fmt = ".")
#plt.plot(T_joined[Cu_lin_selector], model)
#plt.show()

fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,7), layout = "tight",sharex=True, gridspec_kw={'height_ratios': [5, 2]})
ax[0].errorbar(T_joined[Cu_lin_selector], Cu_joined[Cu_lin_selector],Cu_joined_err[Cu_lin_selector], T_joined_err[Cu_lin_selector], fmt = ".", label = "measurements")
ax[0].plot(T_joined[Cu_lin_selector], model, label = "fit")
ax[0].legend(fontsize = 13)
ax[0].title.set_text("Linear regime with Fit, Copper")
ax[0].set_ylabel("$R_{Cu}$ [$\Omega$]")
sigmaRes = np.sqrt(Cu_joined_err[Cu_lin_selector] **2 +(np.gradient(model,T_joined[Cu_lin_selector]) * T_joined_err[Cu_lin_selector])**2)


ax[1].axhline(y=0., color='black', linestyle='--', zorder = 4)
ax[1].errorbar(T_joined[Cu_lin_selector], Cu_joined[Cu_lin_selector]-model,sigmaRes, fmt = ".", label = "residuals")


ax[1].set_ylabel('$R_{Cu}- R_{fit}$ [$\Omega$] ')
ax[1].set_xlabel('$T$ [$°C$] ')
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].legend(fontsize = 13)


fig.text(0.5,0, f'α = ({ m_cu.values["Alpha"]:.8f} +/-{ m_cu.errors["Alpha"]:.8f})'+'$K^{-1}$, $R_0$'+f'= ({ m_cu.values["R_zero"]:.3f} +/-{ m_cu.errors["R_zero"]:.3f} )$\Omega$, chi2/dof = {m_cu.fval:.1f} / {len(Cu_lin_selector - 2)}', horizontalalignment = "center")
fig.subplots_adjust(hspace=0.0)
plt.savefig("linfit_cu.pdf")
plt.show()


#%%linear regime Ta


# fit  R = R_0(1 -alpha*T ), T in C


#join data for he and N
T_joined = np.concatenate((T_N, T_He))
T_joined_err = np.concatenate((Terr_N, Terr_He))

Ta_joined = np.concatenate((Ta_new_N, Ta_new_He))
Ta_joined_err = np.concatenate((sig_Ta_new_N, sig_Ta_new_He))

Ta_lin_selector = Ta_joined>10
    
#cu to T_raum 
T_joined = T_joined - 273.15



chi_lin_Ta = lambda R_zero, Alpha: chi2_lin(lin_approx, T_joined[Ta_lin_selector], Ta_joined[Ta_lin_selector], T_joined_err[Ta_lin_selector], Ta_joined_err[Ta_lin_selector], R_zero, Alpha)


m_ta = iminuit.Minuit(chi_lin_Ta, R_zero = 22.5, Alpha = -0.003)

print(m_ta.migrad())
model = lin_approx(T_joined[Ta_lin_selector], m_ta.values["R_zero"], m_ta.values["Alpha"])
#plt.errorbar(T_joined[Ta_lin_selector], Ta_joined[Ta_lin_selector],Ta_joined_err[Ta_lin_selector], T_joined_err[Ta_lin_selector], fmt = ".")
#plt.plot(T_joined[Ta_lin_selector], model)
#plt.show()

fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,7), layout = "tight",sharex=True, gridspec_kw={'height_ratios': [5, 2]})
ax[0].errorbar(T_joined[Ta_lin_selector], Ta_joined[Ta_lin_selector],Ta_joined_err[Ta_lin_selector], T_joined_err[Ta_lin_selector], fmt = ".", label = "measurements")
ax[0].plot(T_joined[Ta_lin_selector], model, label = "fit")
ax[0].title.set_text("Linear regime with Fit, Copper")
ax[0].set_ylabel("$R_{Ta} [\Omega]$")
ax[0].legend(fontsize = 13)

sigmaRes = np.sqrt(Ta_joined_err[Ta_lin_selector] **2 + (np.gradient(model,T_joined[Ta_lin_selector] )*T_joined_err[Ta_lin_selector])**2)


ax[1].axhline(y=0., color='black', linestyle='--', zorder = 4)
ax[1].errorbar(T_joined[Ta_lin_selector], Ta_joined[Ta_lin_selector]-model,sigmaRes, fmt = ".", label = "residuals")


ax[1].set_ylabel('$R_{Ta}- R_{fit}$ [$\Omega$] ')
ax[1].set_xlabel('$T$ [$°C$] ')
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].legend(fontsize = 13)

fig.text(0.5,0, f'α = ({ m_ta.values["Alpha"]:.8f} +/-{ m_ta.errors["Alpha"]:.8f})'+'$K^{-1}$, $R_0$'+f'= ({ m_ta.values["R_zero"]:.5f} +/-{ m_ta.errors["R_zero"]:.5f} )$\Omega$, chi2/dof = {m_cu.fval:.1f} / {len(Ta_lin_selector - 2)}', horizontalalignment = "center")
fig.subplots_adjust(hspace=0.0)
plt.savefig("ta_linfit.pdf")
plt.show()


#%%non lin fit
def non_lin_approx(t, R_0, alpha, beta):
    return R_0 + alpha*t**beta 

def chi2_nonlin(fit_func, x, y, xerr, yerr, R_0, alpha, beta):
    'chi2 für linearen fit'
    chi2_value = 0
    model = fit_func(x, R_0 = R_0, alpha = alpha, beta=beta)
    chi2_value = np.sum(((y - model) / np.sqrt(yerr**2 + (np.gradient(model, x) * xerr)**2))**2)
    return chi2_value


#%%
#TODO besserer fit


# fit  R = R_0(1 -alpha*T ), T in C


#join data for he and N
T_joined = np.concatenate((T_N, T_He))
T_joined_err = np.concatenate((Terr_N, Terr_He))

Cu_joined = np.concatenate((data_N[:,2], data_He[:,2]))
Cu_joined_err = np.concatenate((sig_Cu_N, sig_Cu_He))


sel = Cu_joined<8
print(sel)

#Cu_lin_selector = Cu_joined>8

chi_nonlin_Cu = lambda R_zero, Alpha, Beta: chi2_nonlin(non_lin_approx, T_joined[sel], Cu_joined[sel], T_joined_err[sel], Cu_joined_err[sel], R_zero, Alpha, Beta)

m_cu_2 = iminuit.Minuit(chi_nonlin_Cu, R_zero = 1, Alpha = 0.003, Beta = 5,)
print(m_cu_2.migrad())


model = non_lin_approx(T_joined[sel], m_cu_2.values["R_zero"], m_cu_2.values["Alpha"], m_cu_2.values["Beta"])

fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,7), layout = "tight",sharex=True, gridspec_kw={'height_ratios': [5, 2]})


ax[0].errorbar(T_joined[sel],Cu_joined[sel],Cu_joined_err[sel],T_joined_err[sel], fmt=  ".", label = "measurements")
ax[0].scatter(T_joined[sel], model, color= "orange", label = "fit")
ax[0].set_ylabel("$R_{Cu}$ [$\Omega$]")
ax[0].legend(fontsize = 13)
ax[0].title.set_text("nonlinear fit, copper")

sigmaRes = np.sqrt(Cu_joined_err[sel] **2 + (np.gradient(model,T_joined[sel] )*T_joined_err[sel])**2)


ax[1].axhline(y=0., color='black', linestyle='--', zorder = 4)
ax[1].errorbar(T_joined[sel], Cu_joined[sel]-model,sigmaRes, fmt = ".", label = "residuals")


ax[1].set_ylabel('$R_{Cu}- R_{fit}$ [$\Omega$] ')
ax[1].set_xlabel('$T$ [$K$] ')
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].legend(fontsize = 13)

fig.text(0.5,0, f'α = ({ m_cu_2.values["Alpha"]:.4f} +/-{ m_cu_2.errors["Alpha"]:.4f})'+'$\Omega *K^{-β}$, $ß$'+f'= ({ m_cu_2.values["Beta"]:.5f} +/-{ m_cu_2.errors["Beta"]:.5f} ), chi2/dof = {m_cu_2.fval:.1f} / {len(sel - 2)}', horizontalalignment = "center")
fig.subplots_adjust(hspace=0.0)
plt.savefig("nonlin_cu.pdf")
plt.show()
#%%ta nonlin fit
#join data for he and N
T_joined = np.concatenate((T_N, T_He))
T_joined_err = np.concatenate((Terr_N, Terr_He))

Ta_joined = np.concatenate((Ta_new_N, Ta_new_He))
Ta_joined_err = np.concatenate((sig_Ta_new_N, sig_Ta_new_He))


sel = np.logical_and(Ta_joined<8, Ta_joined>0.15)
#print(sel)

#Cu_lin_selector = Cu_joined>8

chi_nonlin_Ta = lambda R_zero, Alpha, Beta: chi2_nonlin(non_lin_approx, T_joined[sel], Ta_joined[sel], T_joined_err[sel], Ta_joined_err[sel], R_zero, Alpha, Beta)

m_ta_2 = iminuit.Minuit(chi_nonlin_Ta, R_zero = 1, Alpha = 0.003, Beta = 5,)
print(m_ta_2.migrad())


model = non_lin_approx(T_joined[sel], m_ta_2.values["R_zero"], m_ta_2.values["Alpha"], m_ta_2.values["Beta"])

fig, ax = fig, ax = plt.subplots(2, 1, figsize=(10,7), layout = "tight",sharex=True, gridspec_kw={'height_ratios': [5, 2]})


ax[0].errorbar(T_joined[sel],Ta_joined[sel],Ta_joined_err[sel],T_joined_err[sel], fmt=  ".", label = "measurements")
ax[0].scatter(T_joined[sel], model, color= "orange", label = "fit")
ax[0].set_ylabel("$R_{Cu}$ [$\Omega$]")
ax[0].title.set_text("nonlinear fit, copper")
ax[0].legend(fontsize = 13)


sigmaRes = np.sqrt(Ta_joined_err[sel] **2 + (np.gradient(model,T_joined[sel] )*T_joined_err[sel])**2)


ax[1].axhline(y=0., color='black', linestyle='--', zorder = 4)
ax[1].errorbar(T_joined[sel], Ta_joined[sel]-model,sigmaRes, fmt = ".", label = "residuals")


ax[1].set_ylabel('$R_{Ta}- R_{fit}$ [$\Omega$] ')
ax[1].set_xlabel('$T$ [$°C$] ')
ymax = max([abs(x) for x in ax[1].get_ylim()])
ax[1].set_ylim(-ymax, ymax)
ax[1].legend(fontsize = 13)

fig.text(0.5,0, f'α = ({ m_ta_2.values["Alpha"]:.4f} +/-{ m_ta_2.errors["Alpha"]:.4f})'+'$\Omega *K^{-β}$, $ß$'+f'= ({ m_ta_2.values["Beta"]:.5f} +/-{ m_ta_2.errors["Beta"]:.5f} ), chi2/dof = {m_ta_2.fval:.1f} / {len(sel - 2)}', horizontalalignment = "center")
fig.subplots_adjust(hspace=0.0)
plt.savefig("nonlin_ta.pdf")
plt.show()


#%% si graphs

fig, ax = fig, ax = plt.subplots(3, 1, figsize=(10,15), layout = "tight")

ax[0].errorbar(T_N, data_N[:, 4], sig_Si_N, Terr_N, label = "Resistances Si, N", fmt = ".")
ax[0].errorbar(T_He[Si_filter], data_He[:, 4][Si_filter], sig_Si_He[Si_filter], Terr_He[Si_filter], label = "Resistances Si, He", fmt = ".")
ax[0].set_xlabel("$T [K]$")
ax[0].set_ylabel("$R [\Omega]$")
ax[0].title.set_text("Resistance Measurement Si")
ax[0].legend()

ax[1].errorbar(np.log(T_N), np.log(data_N[:, 4]),1/data_N[:, 4] *  sig_Si_N,1/T_N* Terr_N, label = "Resistances Si, N", fmt = ".")
ax[1].errorbar(np.log(T_He[Si_filter]), np.log(data_He[:, 4][Si_filter]), 1/data_He[:, 4][Si_filter] * sig_Si_He[Si_filter],1/T_He[Si_filter]* Terr_He[Si_filter], label = "Resistances Si, He", fmt = ".")
ax[1].set_xlabel("$ln(T) [ln(K)]$")
ax[1].set_ylabel("$ln(R) [ln(\Omega)]$")
ax[1].title.set_text("Resistance Si, logscale")

#ax[1].set_yscale("log")
#ax[1].set_xscale("log")
ax[1].legend()

ax[2].errorbar(1/T_N, np.log(data_N[:, 4]), 1/data_N[:, 4] * sig_Si_N, 1/T_N**2 * Terr_N, label = "Resistances Si, N", fmt = ".")
ax[2].errorbar(1/T_He[Si_filter], np.log(data_He[:, 4][Si_filter]), 1/data_He[:, 4][Si_filter] * sig_Si_He[Si_filter], 1/T_He[Si_filter]**2 * Terr_He[Si_filter], label = "Resistances Si, He", fmt = ".")
ax[2].set_xlabel("$T^{-1} [K^{-1}]$")
ax[2].set_ylabel("$ln(R) [ln(\Omega)]$")
ax[2].title.set_text("Resistance Si, ln(R), 1/T")
ax[2].legend()

#ax[2].set_yscale("log")
plt.savefig("si_resistances.pdf")


plt.show()

#bonus leitfähigkeit

plt.errorbar(1/T_N, 1/data_N[:, 4],1/data_N[:, 4]**2 *sig_Si_N, 1/T_N**2 * Terr_N, label = "Resistances Si, N", fmt = ".")
plt.errorbar(1/T_He[Si_filter], 1/data_He[:, 4][Si_filter], 1/(data_He[:, 4][Si_filter])**2 *  sig_Si_He[Si_filter], 1/T_He[Si_filter]**2 * Terr_He[Si_filter], label = "Resistances Si, He", fmt = ".")
plt.xlabel("$T^{-1} [K^{-1}]$")
plt.ylabel("$R^{-1} [\Omega^{-1}]$")
plt.title("Conductivity Si")
plt.legend()
plt.yscale("log")
plt.savefig("conductivity_si.pdf")

plt.show()

#plt.errorbar(1/T_N, 1/data_N[:, 4],1/data_N[:, 4]**2 *sig_Si_N, 1/T_N**2 * Terr_N, label = "Resistances Si, N", fmt = ".")
#plt.errorbar(1/T_He[Si_filter], 1/data_He[:, 4][Si_filter], 1/(data_He[:, 4][Si_filter])**2 *  sig_Si_He[Si_filter], 1/T_He[Si_filter]**2 * Terr_He[Si_filter], label = "Resistances Si, He", fmt = ".")
#plt.xlabel("$1/T [1/K]$")
#plt.ylabel("$\sigma [1/\Omega]$")
#plt.title("Resistance Measurement Si")
#plt.legend()
#plt.yscale("log")
#
#plt.show()

plt.errorbar(T_N, data_N[:, 4],sig_Si_N,  Terr_N, label = "Resistances Si, N", fmt = ".")
plt.errorbar(T_He[Si_filter][:225], data_He[:, 4][Si_filter][:225], sig_Si_He[Si_filter][:225], Terr_He[Si_filter][:225], label = "Resistances Si, He", fmt = ".", color = "orange")
plt.xlabel("$T [K]$")
plt.ylabel("$R [\Omega]$")
plt.title("Measurement problems with Si")
plt.legend()
plt.savefig("problems_si.pdf")
#plt.yscale("log")

plt.show()
#%%ta sprungtemp
print(data_He[-100:-40, 1])
plt.scatter(T_c(data_He[-89:-77, 1]), data_He[-89:-77, 3])
plt.axhline(min(data_He[-89:, 3]))
plt.axvline(10)
plt.show()
#TODO Sprungtemperatur