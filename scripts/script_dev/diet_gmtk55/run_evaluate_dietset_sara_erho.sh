#!/bin/bash 
#SBATCH --job-name=test_erho
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH -p long-40core
#SBATCH --mail-type=END
#SBATCH --output=OUT_v_%j.out
#SBATCH --mail-user=Sara.Navarro@stonybrook.edu

cd $HOME
source .bashrc
source modules.sh

conda activate xcq

cd xcquinox/scripts/script_dev/diet_gmtk55
python ./evaluate_dietset_sara.py --load_xnet_path GGA_FxNet_G_d3_n16_s42_erho_10000 --load_cnet_path GGA_FcNet_G_d3_n16_s42_erho_10000 --diet_traj_path ../../script_data/dietgmtkn55-50/diet50.traj --outfile ./test_erho.txt