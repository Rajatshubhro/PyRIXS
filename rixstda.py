'''
Simulate RIXS spectrum using TDDFT-TDA for K & L-edge excited state manifolds.

The RIXS map computation is delegated to rixsci.rixs_map (incoherent) and
rixsci.coherent_rixs_map (coherent Kramers-Heisenberg). This module provides
the PySCF-specific machinery to extract transition dipole moments from TDA
amplitude vectors and assemble the arrays those functions require.

Workflow
--------
1. Run a DFT calculation to obtain MOs (scf.RKS / scf.RHF).
2. Run two TDA calculations: one for core-excited (intermediate) states, one
   for valence-excited (final) states, using select_orbitals_dft to restrict
   the active space.
3. Call build_pyscf_tdm_matrices to extract the TDM 3-vectors and energies.
4. Pass the result to rixs_map (incoherent) or coherent_rixs_map (coherent).

Example -- K-edge RIXS of NH3
------------------------------
    core_slice = [0]   # 1s core orbital
    mftd_n = select_orbitals_dft(mf, coreidx=core_slice, viridx=None)
    tdn = run_tda(mftd_n, nstates=15)

    mftd_f = select_orbitals_dft(mf, coreidx=None, viridx=[10, 12, 14])
    tdf = run_tda(mftd_f, nstates=20)

    mu_fn, mu_ng, en_core_ev, en_final_ev, s_amp_mat = build_pyscf_tdm_matrices(
        mf, tdn, tdf, core_slice=core_slice)

    # Incoherent (independent-intermediate-state) map
    rixs_inc, inc_ax, loss_ax = rixs_map(
        (375, 405), (-1, 15), s_amp_mat, en_core_ev, en_final_ev,
        step_size=0.1, broad_factor=2.4, fwhm=1.2)

    # Coherent (Kramers-Heisenberg) map
    rixs_coh, inc_ax, loss_ax = coherent_rixs_map(
        (375, 405), (-1, 15), mu_fn, mu_ng, en_core_ev, en_final_ev,
        step_size=0.1, broad_factor=2.4, fwhm=1.2)
'''
from pyscf import gto, scf, tddft
import numpy as np
import copy
import sys

from rixsci import rixs_map, coherent_rixs_map

EV_PER_HARTREE = 27.211386245988
HARTREE_PER_EV = 1.0 / EV_PER_HARTREE


def get_tdm_vecs(mf, Xn, Xf, nslice: list = None):
    """
    Compute the TDM 3-vectors <f|mu|n> and <n|mu|g> from PySCF TDA amplitude matrices.

    Parameters
    ----------
    mf     : PySCF SCF object (original full-space calculation)
    Xn     : (n_core_occ, nvirt) TDA amplitude matrix for intermediate (core) state n
    Xf     : (nocc, nvirt_f) TDA amplitude matrix for final (valence) state f
    nslice : list of core orbital indices in the active occupied space;
             if None, all occupied orbitals are used

    Returns
    -------
    fn_amp : (3,) array -- <f|mu|n> Cartesian TDM vector
    ng_amp : (3,) array -- <n|mu|g> Cartesian TDM vector
    """
    nocc = Xf.shape[0]
    nvirt = Xf.shape[1]
    nmo = nocc + nvirt
    if not nslice:
        nslice = list(range(nocc))
        Xn_ncore = nocc
    else:
        Xn_ncore = Xn.shape[0]

    # MO-basis dipole integrals
    ao_dipole = mf.mol.intor('int1e_r')
    mo_dipole = np.einsum('pi,kpq,qj->kij', mf.mo_coeff, ao_dipole, mf.mo_coeff)

    # One-particle TDMs between the two manifolds
    tdm_nf = np.einsum('ia,ib->ab', Xn, Xf[:Xn_ncore, :], optimize=True)
    tdm_fn = np.einsum('ia,ja->ij', Xn, Xf, optimize=True)

    # <n|mu|g>: ground-to-intermediate dipole
    ng_amp = np.einsum('kia,ia->k', mo_dipole[:, nslice, nocc:nmo], Xn, optimize=True)

    # <f|mu|n>: intermediate-to-final dipole (virtual-virtual + occ-occ contributions)
    fn_amp = np.einsum('kba,ba->k', mo_dipole[:, nocc:nmo, nocc:nmo], tdm_nf, optimize=True)
    fn_amp -= np.einsum('kij,ij->k', mo_dipole[:, nslice, :nocc], tdm_fn, optimize=True)

    return fn_amp, ng_amp


def s_amplitudes(mf, Xn, Xf, nslice: list = None):
    """
    Rotationally averaged incoherent scattering amplitude for a single (f, n) pair.

    Uses the crossed Placzek invariant:
        f = (2/15)||S||^2 - (1/30)(|Tr S|^2 + Re[S:S^T])

    Parameters
    ----------
    mf     : PySCF SCF object
    Xn     : TDA amplitude matrix for intermediate (core) state n
    Xf     : TDA amplitude matrix for final (valence) state f
    nslice : list of core orbital indices; if None, use all occupied

    Returns
    -------
    float : rotationally averaged |S_fn|^2 amplitude
    """
    fn_amp, ng_amp = get_tdm_vecs(mf, Xn, Xf, nslice)
    S_fn = np.outer(fn_amp, ng_amp)
    f = (
        (2/15) * np.sum(np.abs(S_fn)**2)
        - (1/30) * (np.abs(np.trace(S_fn))**2 + np.real(np.sum(S_fn * np.conj(S_fn.T))))
    )
    return float(f)


def build_pyscf_tdm_matrices(mf, tdn, tdf, core_slice: list = None):
    """
    Build the TDM arrays required by rixsci.rixs_map and rixsci.coherent_rixs_map
    from PySCF TDA objects.

    The MO-basis dipole integral is computed once and reused for all (f, n) pairs.

    Parameters
    ----------
    mf         : PySCF SCF object (original full-space calculation)
    tdn        : TDA object for core-excited (intermediate) states -- Nn states
    tdf        : TDA object for valence-excited (final) states -- Nf states
    core_slice : list of core orbital indices; if None, use all occupied

    Returns
    -------
    mu_fn       : (Nf, Nn, 3) array -- <val_f|mu|core_n> for all (f, n) pairs
    mu_ng       : (Nn, 3) array     -- <core_n|mu|g> for all core states
    en_core_ev  : (Nn,) array       -- core excitation energies in eV
    en_final_ev : (Nf,) array       -- valence excitation energies in eV
    s_amp_mat   : (Nf, Nn) array    -- incoherent scattering amplitudes for rixs_map
    """
    Nn = tdn.nstates
    Nf = tdf.nstates

    # Pre-compute MO-basis dipole once (avoids redundant AO integral evaluation)
    ao_dipole = mf.mol.intor('int1e_r')
    mo_dipole = np.einsum('pi,kpq,qj->kij', mf.mo_coeff, ao_dipole, mf.mo_coeff)

    # Infer orbital dimensions from the first final state
    Xf_ref = tdf.xy[0][0]
    nocc  = Xf_ref.shape[0]
    nvirt = Xf_ref.shape[1]
    nmo   = nocc + nvirt
    nslice = core_slice if core_slice else list(range(nocc))
    Xn_ncore = tdn.xy[0][0].shape[0]

    mu_fn = np.zeros((Nf, Nn, 3), dtype=float)
    mu_ng = np.zeros((Nn, 3), dtype=float)
    s_amp_mat = np.zeros((Nf, Nn), dtype=float)

    en_core_ev  = np.array([tdn.e[nn] * EV_PER_HARTREE for nn in range(Nn)])
    en_final_ev = np.array([tdf.e[nf] * EV_PER_HARTREE for nf in range(Nf)])

    # <n|mu|g> depends only on Xn
    for nn in range(Nn):
        Xn = tdn.xy[nn][0]
        mu_ng[nn, :] = np.einsum(
            'kia,ia->k', mo_dipole[:, nslice, nocc:nmo], Xn, optimize=True)

    # <f|mu|n> and incoherent amplitude for each (f, n) pair
    for nf in range(Nf):
        Xf = tdf.xy[nf][0]
        for nn in range(Nn):
            Xn = tdn.xy[nn][0]
            tdm_nf = np.einsum('ia,ib->ab', Xn, Xf[:Xn_ncore, :], optimize=True)
            tdm_fn = np.einsum('ia,ja->ij', Xn, Xf, optimize=True)
            fn_amp = np.einsum(
                'kba,ba->k', mo_dipole[:, nocc:nmo, nocc:nmo], tdm_nf, optimize=True)
            fn_amp -= np.einsum(
                'kij,ij->k', mo_dipole[:, nslice, :nocc], tdm_fn, optimize=True)
            mu_fn[nf, nn, :] = fn_amp
            S_fn = np.outer(fn_amp, mu_ng[nn, :])
            s_amp_mat[nf, nn] = float(
                (2/15) * np.sum(np.abs(S_fn)**2)
                - (1/30) * (np.abs(np.trace(S_fn))**2
                            + np.real(np.sum(S_fn * np.conj(S_fn.T))))
            )

    return mu_fn, mu_ng, en_core_ev, en_final_ev, s_amp_mat


def select_orbitals_dft(mf, coreidx: list = None, viridx: list = None):
    """
    Build an active-space SCF object by selecting specific occupied (core) and
    virtual orbital indices.

    Parameters
    ----------
    coreidx : list of occupied orbital indices; if None, use all occupied
    viridx  : list of virtual orbital indices;  if None, use all virtual

    Returns
    -------
    mf_act : copy of mf with mo_coeff/mo_energy/mo_occ restricted to actidx
    """
    if coreidx is None:
        coreidx = np.where(mf.mo_occ == 2)[0].tolist()
    if viridx is None:
        viridx = np.where(mf.mo_occ == 0)[0].tolist()

    actidx = coreidx + viridx
    mf_act = copy.deepcopy(mf)
    mf_act.mo_coeff  = mf.mo_coeff[:, actidx]
    mf_act.mo_energy = mf.mo_energy[actidx]
    mf_act.mo_occ    = mf.mo_occ[actidx]
    mf_act.direct_scf = False
    mf_act._opt = {}
    return mf_act


def run_tda(mf, nstates=1):
    """Run a TDA calculation and return the TDA object."""
    tda = tddft.TDA(mf)
    tda.nstates = nstates
    tda.verbose = 4
    tda.kernel()
    tda.analyze()
    return tda


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

    # -- Step 1: TDA for core-excited (intermediate) states -----------------
    core_slice = [0]   # N 1s core orbital (K-edge)
    mftd_n = select_orbitals_dft(mf, coreidx=core_slice, viridx=None)
    tdn = run_tda(mftd_n, nstates=15)

    # -- Step 2: TDA for valence-excited (final) states ---------------------
    mftd_f = select_orbitals_dft(mf, coreidx=None, viridx=[10, 12, 14])
    tdf = run_tda(mftd_f, nstates=20)

    # -- Step 3: Build TDM arrays (single AO integral call, energies in eV) -
    mu_fn, mu_ng, en_core_ev, en_final_ev, s_amp_mat = build_pyscf_tdm_matrices(
        mf, tdn, tdf, core_slice=core_slice)

    incident_en = (375.0, 405.0)   # eV
    transfer_en = (-1.0,  15.0)    # eV

    # -- Step 4a: Incoherent RIXS map (rixsci.rixs_map) --------------------
    rixs_inc, inc_ax, loss_ax = rixs_map(
        incident_en, transfer_en,
        s_amp_mat, en_core_ev, en_final_ev,
        step_size=0.1, broad_factor=2.4, fwhm=1.2,
    )
    print(f"Incoherent RIXS map shape: {rixs_inc.shape}")

    # -- Step 4b: Coherent (Kramers-Heisenberg) RIXS map -------------------
    rixs_coh, inc_ax, loss_ax = coherent_rixs_map(
        incident_en, transfer_en,
        mu_fn, mu_ng, en_core_ev, en_final_ev,
        step_size=0.1, broad_factor=2.4, fwhm=1.2,
    )
    print(f"Coherent RIXS map shape:    {rixs_coh.shape}")
    print("Passes")
