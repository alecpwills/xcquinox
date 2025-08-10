# Subset 07wf

This subset contains six reaction barriers, two atomization energies, and seven total energies, contained in `subat_ref.traj`.

In this trajectory file, the "reference" energy is stored in the `atom.info` dictionary object under the `target_energy` key. *This energy is in Hartree, the native output unit for PySCF*.

The various `*.dm.npy`, `*.mo_coeff.npy`, `*.mo_occ.npy` are from CCSD(T) calculations, to be used as reference targets or to generate a reference density as needed during training.

If needed, the CCSD(T) total energy is stored in `atom.info['calc_energy']`, in Hartree.

Various other flags in the `atom.info` dictionary are specified, like `name`, `openshell`, `sc` (whether the molecule was used self-consistently in `xcdiff`'s original training), `grid_level`, `sym`, etc.

Trajectory molecules meant to be used for their atomization energies may have the `atomization`, `atomization_ev`, `atomization_H` entries in the corresponding `atom.info` dictionary.

Trajectory molecules meant to be used for their total energies (namely, single atoms) may have had their `calc_energy` and similar entries overwritten to just contain the reference atomic energy, as it is specified in `target_energy`.

Trajectory molecules meant to be used as a reaction barrier have the `reaction` entry, specifying either `reactant` or an integer value. For instance, for the reaction barrier for the hemi-bonded/proton-transfer water dimer, the hemibonded entry has `atom.info['reaction'] = 'reactant'`, and the proton-transfer dimer has `atom.info['reaction'] = 1`, indicating the proton-transfer energy should be subtracted from the energy of the preceding molecule ("1" molecule before it), giving the barrier height as PT - HB. Only the product molecule will have a non-zero `atom.info['target_energy']` or `atom.info['reference_height']`.

## Reaction Barrier

- H + HF -> HF<sub>2</sub>
- OH + CH<sub>3</sub> -> CH<sub>4</sub>O
- H + N<sub>2<sub>O -> HON<sub>2</sub>
- OH + N<sub>2<sub> -> HON<sub>2</sub>
- Li+ -> Li
- The hemibonded water dimer -> proton-transfer water dimer


## Atomization Energy

- The LiF molecule
- The HCl molecule

## Total Energy
- The O atom
- The C atom
- The H atom
- The F atom
- The Cl atom
- The N atom
- The Li atom