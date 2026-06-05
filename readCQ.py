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
        else:
            raise ValueError("Dataset not found")
        
        if "/POSTHF/STATE_ENERGY" in f:
            data_en = f["/POSTHF/STATE_ENERGY"][:]
        else:
            raise ValueError("Dataset not found!!")

    return data_tdm, data_en

def buildCQRIXS(filepath: str)->np.ndarray:
    
    tdmbin, state_enbin = readCQBin(filepath)
    n_pairs = tdmbin.shape[0]
    Imax = int(round((1 + np.sqrt(1 + 8*n_pairs)) / 2))
    
    # Sanity check
    assert Imax * (Imax - 1) // 2 == n_pairs, \
        f"Inverse mapping failed: Imax={Imax}, expected n_pairs={Imax*(Imax-1)//2}, got {n_pairs}"
    assert Imax == len(state_enbin), \
        f"Mismatch: Imax={Imax}, len(state_enbin)={len(state_enbin)}"
    
    tdm_array = np.zeros((Imax, Imax, 3), dtype=complex)
    for I in range(Imax):
        for J in range(I):
            n = (I*(I-1)) // 2 + J
            tdm_array[I, J, :] = tdmbin[n, :]
            tdm_array[J, I, :] = np.conj(tdmbin[n, :])
   
    # Convert to excitation energies (eV)
    state_enbin = (state_enbin - state_enbin[0]) * 27.2114
    
    return tdm_array, state_enbin 
