'''
Simulate RIXS spectrum using TDDFT-TDA for K & L-
Excited State Manifold. User can specify the excitatios that constitute the
excitation manifold.

Example -> (2p-4d RIXS Spectrum of Ruthenium complex)
1. Do one DFT (SCF) calculation to create the MOs.
2. Perform two TDDFT-TDA Calculations, one where you excite
the 2p-orbitals, and the other where you excite the 4d-orbitals.
3. Run the RIXS amp function where you pass the two tddft-tda objects.

'''
from pyscf import gto, scf, tddft
import numpy as np
import copy
import sys

EV_PER_HARTREE = 27.211386245988
HARTREE_PER_EV = 1.0 / EV_PER_HARTREE

'''
Provide Xn and Xf from two separate TD-DFT/TDA calcs
'''
def s_amplitudes(tdm_fn, tdm_ng):

    S_fn =  np.outer(tdm_fn, tdm_ng)
    f = 0.
    f = ((2/15) * np.sum(S_fn**2)) - ((1/30) * ((np.trace(S_fn))**2 + np.sum(S_fn * S_fn.T)))

    return f

def efficient_rixs_map(
    incident_en: tuple[float, float],
    en_transfer: tuple[float, float],
    fn_ampMatrix,      
    en_vec,           
    ef_vec,            
    step_size=0.1,
    broad_factor=2.4,
    fwhm=1.2
):
    if not isinstance(incident_en, tuple) or len(incident_en) != 2:
        raise TypeError("Expected a pair (2-element tuple) for incident energy")
    if not isinstance(en_transfer, tuple) or len(en_transfer) != 2:
        raise TypeError("Expected a pair (2-element tuple) for energy transfer")

    print(f" *** Begin Building RIXS Map *** ")
    alpha_ = 1 / 137.036
    broad_factor_au = broad_factor * HARTREE_PER_EV
    sigma_au = fwhm / (HARTREE_PER_EV * 2 * np.sqrt(2 * np.log(2)))

    # Build energy grids (in a.u.)
    en_iter = np.linspace(*incident_en,  int((incident_en[1]  - incident_en[0])  // step_size)) / EV_PER_HARTREE
    entrans_iter = np.linspace(*en_transfer,  int((en_transfer[1]  - en_transfer[0])  // step_size)) / EV_PER_HARTREE

    en_vec  = np.asarray(en_vec)   # (Nn,)
    ef_vec  = np.asarray(ef_vec)   # (Nf,)

    denom = (en_iter[:, None] - en_vec[None, :])**2 + (broad_factor_au**2) / 4.0  # (Ni, Nn)
    gaussian_broad = np.exp(-0.5 * ((entrans_iter[:, None] - ef_vec[None, :]) / sigma_au)**2)  # (Nj, Nf)
    amp_factor = np.abs(fn_ampMatrix) * ((ef_vec[:, None] * en_vec[None, :] * alpha_)**2)  # (Nf, Nn)

    # --- Contract over (Nf, Nn) for each (Ni, Nj) ---
    # For fixed i,j:
    #   result = sum_{f,n} amp_factor[f,n] / denom[i,n] * gauss[j,f]
    #
    # Factor this as:
    #   result = sum_f gauss[j,f] * sum_n amp_factor[f,n] / denom[i,n]
    #
    # Inner sum: (Nf, Nn) / (Ni, Nn)[broadcast] -> sum over n -> (Ni, Nf)
    residue = np.einsum('fn,in->if', amp_factor, 1.0 / denom, optimize=True)  # (Ni, Nf)
    rixs_intensity_map = np.einsum('if,jf->ij', residue, gaussian_broad, optimize=True)  # (Ni, Nj)
    prefactor = (en_iter[:, None] - entrans_iter[None, :]) / en_iter[:, None]  # (Ni, Nj)

    return prefactor * rixs_intensity_map, en_iter, entrans_iter



if __name__ == "__main__":
    
   
    '''
    1) Run CI calculation and save the tdms in an output or bin file.
    2) Also save the energies of ground state, core and valence excited states.
    3) Read the TDMs and maybe store it like this:
        tdm between state 3 and 4 ---> [3,4,x,y,z] (5 indices: f-state, n-state, x,y,z)
    4) Also save the energy vectors similarly. 

    # Example of doing this:

        s_fnMatrix = np.zeros((ci_f.nstates, ci_n.nstates))
        en_f = np.zeros(tdf.nstates)
        en_n = np.zeros(tdn.nstates)
        for nf in range(ci_f.nstates):
            en_f[nf] = ci_f.e[nf]
            for nn in range(ci_n.nstates):
                
                en_n[nn] = ci_n.e[nn]
                s_fnMatrix[nf,nn] =  s_amplitudes(tdm_fn[nf, nn], tdm_ng[nn,0])

        # Parameters for RIXS Map:
        incident_en = (375, 405)
        transfer_en = (-1, 15)
        rixs_map, incident_en, loss_en = efficient_rixs_map(incident_en, transfer_en, s_fnMatrix, en_n, en_f)
    
    '''
    
    print("Please use it as a module") 
