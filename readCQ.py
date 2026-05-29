'''
Functionality to read in Transition Dipole Moments from Chronus Quantum -
STP-DAS Configuration Interaction calculations.
Reading from bin file (HDF5 format)
'''

import os
import sys
import numpy as np
import h5py

def readCQBin(filepath: str)->np.ndarray:
    with h5py.File(filepath, "r") as f:
        if "/POSTHF/TRANSITION_DIPOLEMOMENTS" in f:
            data_tdm = f["/POSTHF/TRANSITION_DIPOLEMOMENTS"][:]
            print(f"Shape of Transition Dipole Moment: {data.shape})")
        else:
            raise ValueError("Dataset not found")
        
        if "/POSTHF/STATE_ENERGY" in f:
            data_en = f["/POSTHF/STATE_ENERGY"][:]
            print(f"Shape of State Energy in bin file: {data.shape})")
        else:
            raise ValueError("Dataset not found!!")

    return data_tdm, data_en

def buildCQRIXS(filepath: str)->np.ndarray:
    tdmbin, state_enbin = readCQBin(filepath)
    nmax = data.shape[0] - 1
    Imax = int(-0.5 + (np.sqrt(1 + (4*(2*nmax) + 8)) / 2.0)) + 1
    tdm_array = np.zeros((Imax, Imax, 3), dtype=complex)
    for I in range(0, Imax):
        for J in range(0, I):
            n = (I*(I-1)) // 2 + J
            tdm_array[I,J,:] = tdmbin[n,:]

    # Compute the excitation energies for 
    state_enbin = state_enbin[0] - state_enbin

    return tdm_array, state_enbin

