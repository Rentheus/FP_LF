#! /usr/bin/env python3
# -*- coding: utf-8 -*-

### FP Physik
#
# Tobias Sommer, 445306
# Axel Andrée, 422821

files_old = ["LF_He_Ax_To.txt", "LF_noise_Ax_To.txt", "Messung_Fixpunkt_N_Ax_To.txt", "Messung_n_ax_to.txt", "Messungen_Ax_To_raumtemp_noise.txt", "Raumtemperatur.txt"]
files_new = ["LF_He.txt", "LF_Noise.txt", "LF_77K.txt", "LF_N.txt", "LF_Raum_alt.txt", "LF_Raum.txt"]

for i in range(len(files_old)):
    with open(files_old[i]) as f1:
        with open(files_new[i], "w+") as f2:
            f2.write(f1.read().replace(",", "."))
            
