#!/bin/bash 
#SBATCH --job-name=test_v
#SBATCH --output=res.txt
#SBATCH --ntasks=28
#SBATCH --time=12:00:00
#SBATCH -p long-28core
#SBATCH --mail-type=END
#SBATCH --mail-user=Sara.Navarro@stonybrook.edu

module load intel/oneAPI/2022.2
module load compiler mkl mpi
mpirun -n 28 python ./evaluate_dietset_sara.py --load_xnet_path GGA_FxNet_G_d3_n16_s42_v_10000.eqx --load_cnet_path GGA_FcNet_G_d3_n16_s42_v_10000.eqx --diet_traj_path ../../script_data/dietgmtkn55-50/diet50.traj --outfile ./test_v.txt