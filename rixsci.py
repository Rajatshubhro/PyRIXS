import numpy as np

EV_PER_HARTREE = 27.211386245988
HARTREE_PER_EV = 1.0 / EV_PER_HARTREE


def s_amplitudes(tdm_fn, tdm_ng):
    """
    Compute the rotationally averaged incoherent scattering amplitude for a
    single (f, n) pair.

    Parameters
    ----------
    tdm_fn : (3,) array-like  <f|mu|n>  final <- intermediate
    tdm_ng : (3,) array-like  <n|mu|g>  intermediate <- ground

    Returns
    -------
    float : orientationally averaged |S_fn|^2 contribution
    """
    tdm_fn = np.asarray(tdm_fn, dtype=complex)
    tdm_ng = np.asarray(tdm_ng, dtype=complex)

    S_fn = np.outer(tdm_fn.conj(), tdm_ng)          # (3, 3)
    f = (
        (2.0 / 15.0) * np.sum(np.abs(S_fn) ** 2)
        - (1.0 / 30.0) * (
            np.abs(np.trace(S_fn)) ** 2
            + np.real(np.sum(S_fn * np.conj(S_fn.T)))
        )
    )
    return float(f)


def rixs_map(
    incident_en,    # (eV_min, eV_max)
    en_transfer,    # (eV_min, eV_max)
    fn_ampMatrix,   # (Nf, Nn) incoherent scattering amplitudes
    en_vec,         # (Nn,) intermediate state energies in eV
    ef_vec,         # (Nf,) final state energies in eV
    step_size,      # eV
    broad_factor,   # core-hole lifetime FWHM in eV
    fwhm,           # final-state Gaussian FWHM in eV
):
    """Incoherent (independent-intermediate-state) RIXS map."""
    print(" *** Begin Building RIXS Map *** ")
    alpha_ = 1.0 / 137.036
    broad_factor_au = broad_factor * HARTREE_PER_EV
    sigma_au = fwhm * HARTREE_PER_EV / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Use // (floor division) to match the original grid-size convention exactly.
    en_iter      = np.linspace(*incident_en,  int((incident_en[1]  - incident_en[0])  // step_size)) * HARTREE_PER_EV
    entrans_iter = np.linspace(*en_transfer,  int((en_transfer[1]  - en_transfer[0])  // step_size)) * HARTREE_PER_EV

    en_vec = np.asarray(en_vec, dtype=float) * HARTREE_PER_EV   # (Nn,)
    ef_vec = np.asarray(ef_vec, dtype=float) * HARTREE_PER_EV   # (Nf,)

    # Lorentzian denominator:  (omega_in - E_n)^2 + (Gamma/2)^2   (Ni, Nn)
    denom = (en_iter[:, None] - en_vec[None, :]) ** 2 + (broad_factor_au / 2.0) ** 2

    # Gaussian over energy transfer for each final state  (Nj, Nf)
    gaussian_broad = np.exp(
        -0.5 * ((entrans_iter[:, None] - ef_vec[None, :]) / sigma_au) ** 2
    )

    # Scale amplitude matrix by (omega_n * alpha)^2  ->  (Nf, Nn) broadcast over Ni
    amp_factor = fn_ampMatrix * (en_vec[None, :] * alpha_) ** 2   # (Nf, Nn)

    # Sum over n: residue[i, f] = sum_n amp[f,n] / denom[i,n]   (Ni, Nf)
    residue = np.einsum("fn,in->if", amp_factor, 1.0 / denom, optimize=True)

    # Emitted photon energy grid  (Ni, Nj)
    omega_emit = en_iter[:, None] - entrans_iter[None, :]

    # RIXS map  (Ni, Nj) — sum over final states with Gaussian broadening.
    # NOTE: the full prefactor is omega_emit^3 / omega_in (omega_emit^2 here × omega_emit/omega_in
    # below). This differs from the coherent map (omega_emit^1/omega_in). Both match the
    # respective originals; the asymmetry is pre-existing in the source code.
    rixs_intensity_map = (omega_emit ** 2) * np.einsum(
        "if,jf->ij", residue, gaussian_broad, optimize=True
    )
    prefactor = omega_emit / en_iter[:, None]

    return (
        prefactor * rixs_intensity_map,
        en_iter     * EV_PER_HARTREE,
        entrans_iter * EV_PER_HARTREE,
    )


def coherent_rixs_map(
    incident_en,   # (eV_min, eV_max)
    en_transfer,   # (eV_min, eV_max)
    tdm_fn,        # (Nf, Nn, 3)  <f|mu|n>
    tdm_ng,        # (Nn, 3)      <n|mu|g>
    en_vec,        # (Nn,) intermediate state energies in eV
    ef_vec,        # (Nf,) final state energies in eV
    step_size,     # eV
    broad_factor,  # core-hole lifetime FWHM in eV
    fwhm,          # final-state Gaussian FWHM in eV
):
    """Coherent (Kramers-Heisenberg) RIXS map with rotational averaging."""
    print(" *** Begin Building Coherent RIXS Map *** ")
    alpha_ = 1.0 / 137.036
    broad_factor_au = broad_factor * HARTREE_PER_EV
    sigma_au = fwhm * HARTREE_PER_EV / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Use // (floor division) to match the original grid-size convention exactly.
    en_iter      = np.linspace(*incident_en,  int((incident_en[1]  - incident_en[0])  // step_size)) * HARTREE_PER_EV
    entrans_iter = np.linspace(*en_transfer,  int((en_transfer[1]  - en_transfer[0])  // step_size)) * HARTREE_PER_EV

    en_vec = np.asarray(en_vec, dtype=float) * HARTREE_PER_EV   # (Nn,)
    ef_vec = np.asarray(ef_vec, dtype=float) * HARTREE_PER_EV   # (Nf,)

    # Outer product T_fn^{xy} = <f|mu_x|n> * <n|mu_y|g>   (Nf, Nn, 3, 3)
    T_fn = np.einsum("fnx,ny->fnxy", np.conj(tdm_fn), tdm_ng, optimize=True)

    # Scale by (omega_n * alpha)  — applied per intermediate state
    T_fn *= (en_vec[None, :, None, None] * alpha_)

    # Complex Lorentzian denominator  (Ni, Nn)
    gamma_half = broad_factor_au / 2.0
    complex_denom = (en_iter[:, None] - en_vec[None, :]) + 1j * gamma_half

    # Coherent sum over n: A_fg^{xy}   (Ni, Nf, 3, 3)
    A_fg = np.einsum("fnxy,in->ifxy", T_fn, 1.0 / complex_denom, optimize=True)

    # Rotational averaging (Placzek invariants)
    term1 = (2.0 / 15.0) * np.sum(np.abs(A_fg) ** 2, axis=(2, 3))          # (Ni, Nf)
    tr_A  = np.trace(A_fg, axis1=2, axis2=3)                                 # (Ni, Nf)
    term2a = np.abs(tr_A) ** 2                                               # (Ni, Nf)
    A_fg_T = np.swapaxes(A_fg, 2, 3)
    term2b = np.real(np.sum(A_fg * np.conj(A_fg_T), axis=(2, 3)))           # (Ni, Nf)

    averaged_cross_section = term1 - (1.0 / 30.0) * (term2a + term2b)       # (Ni, Nf)

    # Gaussian broadening over final states  (Nj, Nf)
    gaussian_broad = np.exp(
        -0.5 * ((entrans_iter[:, None] - ef_vec[None, :]) / sigma_au) ** 2
    )

    # RIXS map  (Ni, Nj).
    # NOTE: prefactor is omega_emit^1/omega_in — matches the original coherent_rixs_map.
    # The incoherent rixs_map carries omega_emit^3/omega_in; that asymmetry is pre-existing.
    rixs_intensity_map = np.einsum(
        "if,jf->ij", averaged_cross_section, gaussian_broad, optimize=True
    )
    omega_emit = en_iter[:, None] - entrans_iter[None, :]
    final_map = (omega_emit / en_iter[:, None]) * rixs_intensity_map

    return (
        final_map,
        en_iter     * EV_PER_HARTREE,
        entrans_iter * EV_PER_HARTREE,
    )


def doApproxRIXS(
    tdm_array,     # (Nstates, Nstates, 3), indexed [bra, ket, xyz]; state 0 = ground
    excitation_en, # (Nstates,) excitation energies in eV (0 for ground)
    core_list,     # [core_init, core_end)  — half-open slice indices
    valence_list,  # [val_init,  val_end)
    incident_en=None,
    transfer_en=None,
    step_size=0.1,
    broad_factor=1.0,
    fwhm=1.0,
):
    core_init,    core_end    = core_list[0],    core_list[1]
    valence_init, valence_end = valence_list[0], valence_list[1]

    core_exc    = excitation_en[core_init:core_end]      # (Nn,) eV
    valence_exc = excitation_en[valence_init:valence_end] # (Nf,) eV

    if incident_en is None:
        incident_en = (core_exc.min() - 2.0, core_exc.max() + 2.0)
    if transfer_en is None:
        transfer_en = (0.0, valence_exc.max() + 2.0)

    n_core   = core_end   - core_init
    n_final  = valence_end - valence_init

    # FIX 1: pre-allocate with correct shape; fill element-by-element with offset indices.
    s_amp = np.zeros((n_final, n_core), dtype=float)

    for f in range(valence_init, valence_end):
        for n in range(core_init, core_end):
            # FIX 2: <f|mu|n> = tdm_array[f, n, :]  (Cartesian 3-vector)
            #         <n|mu|g> = tdm_array[n, 0, :]  — ground is state index 0, all 3 components
            s_amp[f - valence_init, n - core_init] = s_amplitudes(
                tdm_array[f, n, :],   # <f|mu|n>
                tdm_array[n, 0, :],   # <n|mu|g>  — FIX 3: was [n, 0, 3] (invalid index)
            )

    return rixs_map(
        incident_en, transfer_en,
        s_amp, core_exc, valence_exc,
        step_size, broad_factor, fwhm,
    )


def doCoherentRIXS(
    tdm_array,     # (Nstates, Nstates, 3), indexed [bra, ket, xyz]; state 0 = ground
    excitation_en, # (Nstates,) excitation energies in eV (0 for ground)
    core_list,     # [core_init, core_end)
    valence_list,  # [val_init,  val_end)
    incident_en=None,
    transfer_en=None,
    step_size=0.1,
    broad_factor=1.0,
    fwhm=1.0,
):
    core_init,    core_end    = core_list[0],    core_list[1]
    valence_init, valence_end = valence_list[0], valence_list[1]

    core_exc    = excitation_en[core_init:core_end]
    valence_exc = excitation_en[valence_init:valence_end]

    if incident_en is None:
        incident_en = (core_exc.min() - 2.0, core_exc.max() + 2.0)
    if transfer_en is None:
        transfer_en = (0.0, valence_exc.max() + 2.0)

    # FIX 4: <f|mu|n> — rows = valence (final), cols = core (intermediate)
    #         <n|mu|g> — rows = core (intermediate), col 0 = ground state
    fn_tdm = tdm_array[valence_init:valence_end, core_init:core_end, :]  # (Nf, Nn, 3)
    ng_tdm = tdm_array[core_init:core_end, 0, :]                          # (Nn, 3)

    return coherent_rixs_map(
        incident_en, transfer_en,
        fn_tdm, ng_tdm,
        core_exc, valence_exc,
        step_size, broad_factor, fwhm,
    )
