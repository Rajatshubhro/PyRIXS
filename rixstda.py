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
def s_amplitudes(mf, Xn, Xf, nslice: list = None):

    nocc = Xf.shape[0]
    nvirt = Xf.shape[1]
    nmo = nocc + nvirt
    if (not nslice):
        nslice = [i for i in range(nocc)]
        Xn_ncore = nocc
    else:
        Xn_ncore = Xn.shape[0]

    # MO-basis dipole
    ao_dipole = mf.mol.intor('int1e_r')
    mo_dipole = np.einsum('pi,kpq,qj->kij', mf.mo_coeff, ao_dipole, mf.mo_coeff)
    
    # Get the tdms:
    tdm_nf = np.einsum('ia,ib->ab', Xn, Xf[:Xn_ncore, :], optimize=True)
    tdm_fn = np.einsum('ia,ja->ij', Xn, Xf, optimize=True)
    # Amplitudes
    ng_amp = np.einsum('kia,ia->k', mo_dipole[:, nslice, nocc : nmo], Xn, optimize=True)
    fn_amp = np.einsum('kba,ba->k', mo_dipole[:, nocc : nmo, nocc : nmo], tdm_nf, optimize=True)
    fn_amp -= np.einsum('kij,ij->k', mo_dipole[:, nslice, :nocc], tdm_fn, optimize=True)

    # Obtain S_fn matrix:
    S_fn =  np.outer(fn_amp, ng_amp)
    f = 0.
    f = ((2/15) * np.sum(S_fn**2)) - ((1/30) * ((np.trace(S_fn))**2 + np.sum(S_fn * S_fn.T)))

    return f

''' 
Provide ranges for:
incident_en --> (500, 520)
en_tranfer --> (-2, 15)
broad_factor, fwhm --> Optional args
Defaults:
    step_size = 0.1 eV
    broad_factor = 2.4 eV
    fwhm = 1.2 eV
'''
def rixs_map(incident_en: tuple[float, float], en_transfer: tuple[float, float], fn_ampMatrix, en_vec, ef_vec, step_size = 0.1, broad_factor = 2.4, fwhm = 1.2):

    if not isinstance(incident_en, tuple) or len(incident_en) != 2:
        raise TypeError("Expected a pair (2-element tuple) for incident energy")
    if not isinstance(en_transfer, tuple) or len(en_transfer) != 2:
        raise TypeError("Expected a pair (2-element tuple) for energy transfer")
   
    alpha_ = 1 / 137.036
    broad_factor /= 27.2114
    sigma_ev = fwhm / (27.2114 * 2 * np.sqrt(2 * np.log(2)))
    gaussian_broadening = lambda en_transfer, ef: np.exp(-0.5 * ((en_transfer - ef) / sigma_ev)**2)

    init_en, final_en = incident_en
    nsteps = int((final_en - init_en) // step_size)
    en_iter = np.linspace(init_en, final_en, nsteps)
    en_iter /= 27.2114 # Conversion to a.u

    init_entrans, final_entrans = en_transfer
    nsteps = int((final_entrans - init_entrans) // step_size)
    entrans_iter = np.linspace(init_entrans, final_entrans, nsteps)
    entrans_iter /= 27.2114 # Conversion to a.u

    rixs_intensity_map = np.zeros((len(en_iter), len(entrans_iter)))

    for i, en_inc in enumerate(en_iter):
        for j, en_trans in enumerate(entrans_iter):
            
            en_emit = en_inc - en_trans
            prefactor = en_emit / en_inc

            result = 0.
            for nf, en_f in enumerate(ef_vec):
                gauss_bf = gaussian_broadening(en_trans, en_f)
                for nn, en_n  in enumerate(en_vec):
                    
                    denom = ((en_inc - en_n)**2) + ((broad_factor**2) / 4.)
                    result += np.abs(fn_ampMatrix[nf, nn]) * (((en_f * en_n * alpha_)**2) / denom) * gauss_bf

            rixs_intensity_map[i,j] = result

    return rixs_intensity_map,

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
coreidx and viridx are optional args.
If set to None, they include all the core / virt orbs respectively.
Example:
    coreidx --> [0,1,2,3]
    viridx  --> [6,7,8,9]
'''
def select_orbitals_dft(mf, coreidx: list = None, viridx: list = None):
    
    if (coreidx == None):
        coreidx = np.where(mf.mo_occ == 2)[0].tolist()
    if (viridx == None):
        viridx = np.where(mf.mo_occ == 0)[0].tolist()
            
    actidx = coreidx + viridx
    mf_act = copy.deepcopy(mf)
    mf_act.mo_coeff = mf.mo_coeff[:,actidx]
    mf_act.mo_energy = mf.mo_energy[actidx]
    mf_act.mo_occ = mf.mo_occ[actidx]
    mf_act.direct_scf = False
    mf_act._opt = {}

    return mf_act

'''
Easy Run TD-DFT TDA:
nstates in optional parameter

'''
def run_tda(mf, nstates=1):

    tda = tddft.TDA(mf)
    tda.nstates = nstates
    tda.verbose = 4
    tda.kernel()
    tda.analyze()
    
    return tda


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
            s_fnMatrix[nf,nn] =  s_amplitudes(mf, Xn, Xf, core_slice)
            en_n[nn] = tdn.e[nn]

    # Parameters for RIXS Map:
    incident_en = (375, 405)
    transfer_en = (-1, 15)
    rixs_map, incident_en, loss_en = efficient_rixs_map(incident_en, transfer_en, s_fnMatrix, en_n, en_f)
    print("Passes")
