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
sig_C_He = statsigma_He(Intervalle_He, data_He[:,1])
sig_Cu_He = statsigma_He(Intervalle_He, data_He[:,2])
sig_Ta_He = statsigma_He(Intervalle_He, data_He[:,3], plot = "nein")
sig_Si_He = statsigma_He(Intervalle_He, data_He[:,4])

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
ax[0].legend()

ax[1].errorbar(T_N, data_N[:, 1], sig_C_N, Terr_N, label = "Resistances C, N", fmt = ".")
ax[1].errorbar(T_He, data_He[:, 1], sig_C_He, Terr_He, label = "Resistances C, He", fmt = ".")
ax[1].set_xlabel("$T [K]$")
ax[1].set_ylabel("$R [\Omega]$")
ax[1].legend()

ax[2].errorbar(T_N, data_N[:, 2], sig_Cu_N, Terr_N, label = "Resistances Cu, N", fmt = ".")
ax[2].errorbar(T_He, data_He[:, 2], sig_Cu_He, Terr_He, label = "Resistances Cu, He", fmt = ".")
ax[2].set_xlabel("$T [K]$")
ax[2].set_ylabel("$R [\Omega]$")
ax[2].legend()

ax[3].errorbar(T_N, data_N[:, 3], sig_Ta_N, Terr_N, label = "Resistances Ta, N", fmt = ".")
ax[3].errorbar(T_He, data_He[:, 3], sig_Ta_He, Terr_He, label = "Resistances Ta, He", fmt = ".")
ax[3].set_xlabel("$T [K]$")
ax[3].set_ylabel("$R [\Omega]$")
ax[3].legend()

ax[4].errorbar(T_N, data_N[:, 4], sig_Si_N, Terr_N, label = "Resistances Si, N", fmt = ".")
ax[4].errorbar(T_He[Si_filter], data_He[Si_filter][:, 4], sig_Si_He[Si_filter], Terr_He[Si_filter], label = "Resistances Si, He", fmt = ".")
ax[4].set_xlabel("$T [K]$")
ax[4].set_ylabel("$R [\Omega]$")
ax[4].legend()

plt.show()
#%% Tantalum temp


fig, ax = fig, ax = plt.subplots(2, 1, figsize=(9,7), layout = "tight")

ax[0].errorbar(T_N, data_N[:,3], sig_Ta_N, Terr_N,  label= "Measured Tantalum Resistances, N")
ax[0].errorbar(T_He, data_He[:,3], sig_Ta_He, Terr_He, label= "Measured Tantalum Resistances, He")


Ta_small_filter = data_He[:,3]<1

Ta_small = data_He[:,3][data_He[:,3]<1]
T_Ta_small = T_He[Ta_small_filter]
ax[1].errorbar(T_Ta_small, Ta_small,np.array(sig_Ta_He)[Ta_small_filter], Terr_He[Ta_small_filter],  label= "Measured Tantalum Resistance")

plt.show()

#%%ta sprungtemp
print(data_He[-100:-40, 1])
plt.scatter(T_c(data_He[-89:-77, 1]), data_He[-89:-77, 3])
plt.axhline(min(data_He[-89:, 3]))
plt.axvline(10)
plt.show()
