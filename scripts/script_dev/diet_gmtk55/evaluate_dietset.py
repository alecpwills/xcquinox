import argparse
import pyscf
import pyscfad
from xcquinox.pyscf import eval_xc_gga_pol
from xcquinox.utils import gen_grid_s, PBE_Fx, PBE_Fc, calculate_stats, lda_x, pw92c_unpolarized
from xcquinox.train import Pretrainer, Optimizer
from xcquinox.loss import compute_loss_mae
from xcquinox import net, xc
from ase.io import read
import jax.numpy as jnp
import jax
import equinox as eqx
import optax
import numpy as np
from pyscf import dft, scf, gto, cc
from pyscfad import dft as dft_ad
from pyscfad import gto as gto_ad
from pyscfad import scf as scf_ad
from functools import partial
import pylibxc
import pyscfad.dft as dftad
from jax import custom_jvp
jax.config.update("jax_enable_x64", True)  # Enables 64 bit precision

if __name__ == '__main__':
    # parse script arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--diet_traj_path', action='store',
                        help='Path to .xyz/.traj file containing list of configurations')
    parser.add_argument('--outfile', action='store', default='output.txt',
                        help='Filename for the output results. This gets OVERWRITTEN with each script call if the name is not changed.')
    parser.add_argument('--load_xnet_path', action='store',
                        help='Path to .eqx checkpoint file of the desired exchange network to use.', default=None)
    parser.add_argument('--load_cnet_path', action='store',
                        help='Path to .eqx checkpoint file of the desired exchange network to use.', default=None)
    parser.add_argument('--gennet_depth', default=3, type=int, action='store',
                        help='Depth of the random X/C network to initialize for calculation. This is only used if the X/C network is not loaded in manually.')
    parser.add_argument('--gennet_nodes', default=16, type=int, action='store',
                        help='Nodes/layer of the random X/C network to initialize for calculation. This is only used if the X/C network is not loaded in manually.')
    parser.add_argument('--gennet_seed', default=92017, type=int, action='store',
                        help='Random seed to use in weight initialization of the random X/C network to initialize for calculation. This is only used if the X/C network is not loaded in manually.')
    parser.add_argument('--calc_gridlevel', default=1, type=int, action='store',
                        help='Grid level to specify for the calculation. 1 is coarser, 9 is finer.')
    parser.add_argument('--calc_maxscf', default=25, type=int, action='store',
                        help='Maximum number of SCF steps to allow in the calculation. Used if non-converging networks take up too much resources.')
    parser.add_argument('--calc_maxmem', default=32000, type=int, action='store',
                        help='Maximum memory to tell PySCF is available for a given calculation (in MB).')
    args = parser.parse_args()

    if args.load_xnet_path:
        xnet = net.load_xcquinox_model(args.load_xnet_path)
    else:
        xnet = net.GGA_FxNet_sigma(depth=args.gennet_depth,
                                   nodes=args.gennet_nodes,
                                   seed=args.gennet_seed)
    if args.load_cnet_path:
        cnet = net.load_xcquinox_model(args.load_cnet_path)
    else:
        cnet = net.GGA_FcNet_sigma(depth=args.gennet_depth,
                                   nodes=args.gennet_nodes,
                                   seed=args.gennet_seed)
    xc = xc.RXCModel_GGA(xnet=xnet, cnet=cnet)

    OVERWRITE_EVAL_XC = partial(eval_xc_gga_pol, xcmodel=xc)
    GRID_LEVEL = args.calc_gridlevel
    MAX_SCF_STEPS = args.calc_maxscf

    traj = read(args.diet_traj_path, ':')
    print("Trajectory loaded in.")
    print("Printing information...")
    for idx, at in enumerate(traj):
        print(20*'=')
        print(idx, at)
        print(at.info)
        print(at.get_chemical_symbols())
        print(at.positions)

    results = {idx: 0 for idx in range(len(traj))}
    with open(args.outfile, 'w') as f:
        f.write("atidx\tatsymbols\tatformula\tsubset\tsubsetind\tspecies\tcount\trefweight\trefen\tcalcen\tcalcconv\n")
    for idx, sys in enumerate(traj):
        atstr = ''
        for aidx, sysat in enumerate(sys.get_chemical_symbols()):
            atstr += f"{sysat} {sys.positions[aidx][0]} {sys.positions[aidx][1]} {sys.positions[aidx][2]}\n"
        mol = gto_ad.Mole(atom=atstr, charge=sys.info.get('charge', 0), spin=sys.info.get('spin', 0))
        mol.build()
        # If the local memory usage reaches this max_memory value, the SCF cycles are broken down into sub-loops over small sections of the grid that take *forever* to get through
        mol.max_memory = args.calc_maxmem
        print("Beginning calculation...")
        print(f"{idx} -- {sys.symbols}/{sys.get_chemical_formula()}")
        # if sys.get_chemical_formula() != 'H':
        if sys.info.get('spin', 0) == 0:
            print("SPIN 0 -> RKS")
            mf = dft_ad.RKS(mol)
            mf.grids.level = GRID_LEVEL
            mf.max_cycle = MAX_SCF_STEPS
            mf.define_xc_(OVERWRITE_EVAL_XC, 'GGA')
            mf.kernel()
        else:
            print("NONZERO SPIN -> UKS")
            mf = dft_ad.UKS(mol)
            mf.grids.level = GRID_LEVEL
            mf.max_cycle = MAX_SCF_STEPS
            mf.define_xc_(OVERWRITE_EVAL_XC, 'GGA')
            mf.kernel()
        results[idx] = (mf.e_tot, mf.converged)
        print(f"Results: CONVERGED = {mf.converged}, ENERGY = {mf.e_tot}")
        with open(args.outfile, 'a') as f:
            f.write(
                f"{idx}\t{sys.symbols}\t{sys.get_chemical_formula()}\t{sys.info['subset']}\t{sys.info['subsetind']}\t{sys.info['species']}\t{sys.info['count']}\t{sys.info['refweight']}\t{sys.info['refen']}\t{mf.e_tot}\t{mf.converged}\n")
