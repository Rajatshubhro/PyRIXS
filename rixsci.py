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

'''
if "main" : Runs test on RIXS
'''
if __name__ == "__main__":
    
    np.set_printoptions(threshold=sys.maxsize, linewidth=80)
    mol = gto.Mole()
    mol.atom = '''
    N  0.000000  0.000000  0.000000
    H  0.000000  0.937700  0.381600
    H  0.812100 -0.468800  0.381600
    H -0.812100 -0.468800  0.381600
    '''
    mol.basis = 'cc-pVDZ'
    mol.verbose = 4
    mol.build()
    
    mf = scf.RKS(mol)
    mf.verbose = 4
    mf.xc = 'b3lyp'
    mf.kernel()
    mf.analyze()
    
    # Select core and virtual orbitals for TDDFT:
    # mftd_n: Evaluate core-excited states
    # mftd_f: Obtain the valence-excited states
    '''
    n_slice: List of ncore orbitals that is user-defined for 
    the core excited states.
    '''
    core_slice = [0] # K-edge Core excited state
    mftd_n = select_orbitals_dft(mf, coreidx = core_slice, viridx = None)
    tdn = run_tda(mftd_n, 15)
    tdn.analyze()
    mftd_f = select_orbitals_dft(mf, coreidx = None, viridx=[10,12,14])
    tdf = run_tda(mftd_f, 20)
    tdf.analyze()
 

    # Make the S_fn Matrix:
    s_fnMatrix = np.zeros((tdf.nstates, tdn.nstates))
    en_f = np.zeros(tdf.nstates)
    en_n = np.zeros(tdn.nstates)
    for nf in range(tdf.nstates):
        en_f[nf] = tdf.e[nf]
        for nn in range(tdn.nstates):

            Xn = tdn.xy[nn][0]
            Xf = tdf.xy[nf][0]
            s_fnMatrix[nf,nn] =  s_amplitudes(mf, tdm_fn, tdm_ng)
            en_n[nn] = tdn.e[nn]

    # Parameters for RIXS Map:
    incident_en = (375, 405)
    transfer_en = (-1, 15)
    rixs_map, incident_en, loss_en = efficient_rixs_map(incident_en, transfer_en, s_fnMatrix, en_n, en_f)
    print("Passes")
