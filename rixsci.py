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
import numpy as np
import copy
import sys

EV_PER_HARTREE = 27.211386245988
HARTREE_PER_EV = 1.0 / EV_PER_HARTREE

'''
Provide Xn and Xf from two separate TD-DFT/TDA calcs
'''
def s_amplitudes(tdm_fn, tdm_ng):

    S_fn =  np.outer(tdm_fn.conj(), tdm_ng)
    f = (((2/15) * np.sum(np.abs(S_fn)**2)) - 
        ((1/30) * (np.abs((np.trace(S_fn)))**2 + 
        np.real(np.sum(S_fn * np.conj(S_fn.T))))))

    return f

def rixs_map(
    incident_en: tuple[float, float],
    en_transfer: tuple[float, float],
    fn_ampMatrix,      
    en_vec,           
    ef_vec,            
    step_size,
    broad_factor,
    fwhm
):
    if not isinstance(incident_en, tuple) or len(incident_en) != 2:
        raise TypeError("Expected a pair (2-element tuple) for incident energy")
    if not isinstance(en_transfer, tuple) or len(en_transfer) != 2:
        raise TypeError("Expected a pair (2-element tuple) for energy transfer")

    print(f" *** Begin Building RIXS Map *** ")
    alpha_ = 1 / 137.036
    broad_factor_au = broad_factor / EV_PER_HARTREE
    sigma_au = fwhm / (EV_PER_HARTREE * 2 * np.sqrt(2 * np.log(2)))

    # Build energy grids (in a.u.)
    en_iter = np.linspace(*incident_en,  int((incident_en[1]  - incident_en[0])  // step_size)) / EV_PER_HARTREE
    entrans_iter = np.linspace(*en_transfer,  int((en_transfer[1]  - en_transfer[0])  // step_size)) / EV_PER_HARTREE

    en_vec  = np.asarray(en_vec) / EV_PER_HARTREE   # (Nn,)
    ef_vec  = np.asarray(ef_vec) / EV_PER_HARTREE  # (Nf,)

 # --- DIAGNOSTICS ---
    print(f"en_iter (a.u.):      {en_iter.min():.4f} to {en_iter.max():.4f}")
    print(f"entrans_iter (a.u.): {entrans_iter.min():.4f} to {entrans_iter.max():.4f}")
    print(f"en_vec (a.u.):       {en_vec.min():.4f} to {en_vec.max():.4f}")
    print(f"ef_vec (a.u.):       {ef_vec.min():.4f} to {ef_vec.max():.4f}")
    print(f"broad_factor_au:     {broad_factor_au:.6f}")
    print(f"sigma_au:            {sigma_au:.6f}")
    print(f"fn_ampMatrix max:    {np.abs(fn_ampMatrix).max():.6e}")
    print(f"fn_ampMatrix nonzero: {np.count_nonzero(fn_ampMatrix)}")


    denom = (en_iter[:, None] - en_vec[None, :])**2 + (broad_factor_au**2) / 4.0
    print(f"\ndenom min: {denom.min():.6e}, max: {denom.max():.6e}")
    print(f"1/denom min: {(1/denom).min():.6e}, max: {(1/denom).max():.6e}")

    gaussian_broad = np.exp(-0.5 * ((entrans_iter[:, None] - ef_vec[None, :]) / sigma_au)**2)
    print(f"gaussian_broad min: {gaussian_broad.min():.6e}, max: {gaussian_broad.max():.6e}")
    print(f"gaussian_broad nonzero: {np.count_nonzero(gaussian_broad > 1e-10)}")

    amp_factor = np.abs(fn_ampMatrix) * ((en_vec[None, :] * alpha_)**2)
    print(f"\namp_factor min: {amp_factor.min():.6e}, max: {amp_factor.max():.6e}")
    print(f"amp_factor nonzero: {np.count_nonzero(amp_factor)}")

    residue = np.einsum('fn,in->if', amp_factor, 1.0 / denom, optimize=True)
    print(f"\nresidue min: {residue.min():.6e}, max: {residue.max():.6e}")
    print(f"residue nonzero: {np.count_nonzero(residue)}")
    
    omega_emit = en_iter[:, None] - entrans_iter[None, :]   # (Ni, Nj)
    rixs_intensity_map = (omega_emit**2) * np.einsum('if,jf->ij', residue, gaussian_broad, optimize=True)
    prefactor = omega_emit / en_iter[:, None]
    print(f"\nrixs_intensity_map min: {rixs_intensity_map.min():.6e}, max: {rixs_intensity_map.max():.6e}")
    print(f"prefactor min: {prefactor.min():.6e}, max: {prefactor.max():.6e}")
    return prefactor * (rixs_intensity_map), en_iter * EV_PER_HARTREE, entrans_iter * EV_PER_HARTREE


def coherent_rixs_map(
    incident_en: tuple[float, float],
    en_transfer: tuple[float, float],
    tdm_fn,  # Shape: (Nf, Nn, 3) - Cartesian transition dipoles <f|mu|n>
    tdm_ng,  # Shape: (Nn, 3)    - Cartesian transition dipoles <n|mu|g>
    en_vec,  # Shape: (Nn,)      - Intermediate state energies
    ef_vec,  # Shape: (Nf,)      - Final state energies
    step_size,
    broad_factor,
    fwhm
):
    if not isinstance(incident_en, tuple) or len(incident_en) != 2:
        raise TypeError("Expected a pair (2-element tuple) for incident energy")
    if not isinstance(en_transfer, tuple) or len(en_transfer) != 2:
        raise TypeError("Expected a pair (2-element tuple) for energy transfer")

    print(f" *** Begin Building Coherent RIXS Map *** ")
    alpha_ = 1 / 137.036
    broad_factor_au = broad_factor / EV_PER_HARTREE
    sigma_au = fwhm / (EV_PER_HARTREE * 2 * np.sqrt(2 * np.log(2)))

    # Build energy grids (in a.u.)
    en_iter = np.linspace(*incident_en,  int((incident_en[1]  - incident_en[0])  // step_size)) / EV_PER_HARTREE
    entrans_iter = np.linspace(*en_transfer,  int((en_transfer[1]  - en_transfer[0])  // step_size)) / EV_PER_HARTREE

    en_vec  = np.asarray(en_vec) / EV_PER_HARTREE   # (Nn,)
    ef_vec  = np.asarray(ef_vec) / EV_PER_HARTREE   # (Nf,)

    # 1. Build the Numerator Tensors (Nf, Nn, 3, 3)
    # T_fn^{ij} = <f|mu_i|n> * <n|mu_j|g>
    print("Building intermediate tensor outer products...")
    T_fn = np.einsum('fnx,ny->fnxy', tdm_fn.conj(), tdm_ng, optimize=True)
    
    # Apply the (omega_n * alpha)^2 prefactor to the tensor immediately
    # Note: we use en_vec (omega_n) to match your previous setup
    T_fn *= (en_vec[None, :, None, None] * alpha_)
    
    # 2. Build the Complex Energy Denominator (Ni, Nn)
    # Denom = (omega_in - E_n) + i(\Gamma / 2)
    print("Evaluating complex energy denominators...")
    gamma_half = broad_factor_au / 2.0
    complex_denom = (en_iter[:, None] - en_vec[None, :]) + 1j * gamma_half

    # 3. The Coherent Sum over intermediate states 'n' (Ni, Nf, 3, 3)
    # A_fg = Sum_n [ T_fn / complex_denom ]
    print("Summing coherent amplitudes over intermediate states...")
    A_fg = np.einsum('fnxy,in->ifxy', T_fn, 1.0 / complex_denom, optimize=True)

    # 4. Rotational Averaging (Vectorized Placzek Invariants)
    # Yields the orientationally averaged cross section matrix (Ni, Nf)
    print("Applying rotational averaging invariants...")
    
    # Term 1: (2/15) * sum_{ij} |A_ij|^2
    term1 = (2.0 / 15.0) * np.sum(np.abs(A_fg)**2, axis=(2, 3))
    
    # Term 2a: |Tr(A)|^2
    tr_A = np.trace(A_fg, axis1=2, axis2=3) 
    term2a = np.abs(tr_A)**2
    
    # Term 2b: sum_{ij} A_ij * (A_ji)*
    A_fg_T = np.swapaxes(A_fg, 2, 3) # Transpose spatial dimensions
    term2b = np.real(np.sum(A_fg * np.conj(A_fg_T), axis=(2, 3)))
    
    # Combine terms according to your specific rotational invariants
    averaged_cross_section = term1 - (1.0 / 30.0) * (term2a + term2b)
    
    # 5. Emission Energy & Final State Broadening
    print("Applying final state Gaussian broadening...")
    gaussian_broad = np.exp(-0.5 * ((entrans_iter[:, None] - ef_vec[None, :]) / sigma_au)**2)
    rixs_intensity_map = np.einsum('if,jf->ij', averaged_cross_section, gaussian_broad, optimize=True)
    omega_emit = en_iter[:, None] - entrans_iter[None, :]
    final_map = (omega_emit / en_iter[:, None]) * rixs_intensity_map   
    
    print(f"T_fn max:      {np.abs(T_fn).max():.6e}")
    print(f"1/denom max:   {(1/np.abs(complex_denom)).max():.6e}")
    print(f"A_fg max (after coherent sum + ω_emit + α): {np.abs(A_fg).max():.6e}")
    print(f"|A_fg|² max:   {(np.abs(A_fg)**2).max():.6e}")
    print(f"term1 max:     {term1.max():.6e}")
    print(f"term2a max:    {term2a.max():.6e}")
    print(f"term2b max:    {term2b.max():.6e}")
    print(f"averaged_cross_section max: {averaged_cross_section.max():.6e}")
    print(f"gaussian_broad max: {gaussian_broad.max():.6e}")
    print(f"rixs_intensity_map max: {rixs_intensity_map.max():.6e}")
    print(f"final_map max: {final_map.max():.6e}")
    print(f"tdm_fn max: {np.abs(tdm_fn).max():.6e}")
    print(f"tdm_ng max: {np.abs(tdm_ng).max():.6e}")

    return final_map, en_iter * EV_PER_HARTREE, entrans_iter * EV_PER_HARTREE


def doApproxRIXS(tdm_array, excitation_en,
        core_list, valence_list,
        incident_en=None, transfer_en=None,
        step_size=0.1,
        broad_factor=1.0,
        fwhm=1.0):

    core_init, core_end = core_list[0], core_list[1]
    valence_init, valence_end = valence_list[0], valence_list[1]
    core_exc = excitation_en[core_init : core_end]
    valence_exc = excitation_en[valence_init : valence_end]

    if incident_en is None:
        incident_en = (core_exc.min() - 2, core_exc.min() + 100)
    if transfer_en is None:
        transfer_en = (-10, valence_exc.max() + 2)

    s_amp = np.zeros((len(valence_exc), len(core_exc)))
    for f in range(valence_init, valence_end):
        for n in range(core_init, core_end):
            s_amp = s_amplitudes(tdm_array[f,n,:], tdm_array[n,0,3])

    return rixs_map(incident_en, transfer_en, s_amp, core_exc, 
            valence_exc, step_size, broad_factor, fwhm)


def doCoherentRIXS(tdm_array, excitation_en, 
        core_list, valence_list, 
        incident_en = None, transfer_en = None,
        step_size = 0.1, 
        broad_factor = 1.0, 
        fwhm = 1.0):

    core_init, core_end = core_list[0], core_list[1]
    valence_init, valence_end = valence_list[0], valence_list[1]
    core_exc = excitation_en[core_init : core_end]
    valence_exc = excitation_en[valence_init : valence_end]

    if incident_en is None:
        incident_en = (core_exc.min() - 2, core_exc.min() + 100)
    if transfer_en is None:
        transfer_en = (-10, valence_exc.max() + 2)

    # Create tdms for <f|n> and <n|g>:
    fn_tdm = tdm_array[valence_init:valence_end, core_init:core_end,:]
    ng_tdm = tdm_array[core_init:core_end, 0, :]
    
    return coherent_rixs_map(incident_en, transfer_en, fn_tdm, ng_tdm,
            core_exc, valence_exc, step_size, broad_factor, fwhm)


if __name__ == "__main__":
    
    print("Please use it as a module")
