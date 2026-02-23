Uses TDDFT-TDA in PySCF to simulate RIXS map for any type of core-excitations. Details on running coming soon.

Example usage script:

    import sys
    import numpy as np
    from pyscf import gto, scf, tddft
    from rixs import rixstda

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
    mftd_n = rixstda.select_orbitals_dft(mf, core_slice)
    tdn = rixstda.run_tda(mftd_n, 15)
    tdn.analyze()
    mftd_f = rixstda.select_orbitals_dft(mf)
    tdf = rixstda.run_tda(mftd_f, 20)
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
            s_fnMatrix[nf,nn] =  rixstda.s_amplitudes(mf, Xn, Xf, core_slice)
            en_n[nn] = tdn.e[nn]


    # Parameters for RIXS Map:
    incident_en = (375, 405)
    transfer_en = (-1, 15)

    np.set_printoptions(precision=20, suppress=True)
    rixs_map = rixstda.rixs_map(incident_en, transfer_en, s_fnMatrix, en_n, en_f)
    print(f"RIXS MAP: \n {rixs_map} \n\n Incident en:\n {incident_en} \n\n En transfer: \n {transfer_en}")
