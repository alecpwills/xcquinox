#!/bin/bash 
#SBATCH --job-name=test_train
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH -p long-40core
#SBATCH --mail-type=END,FAIL
#SBATCH --output=OUT_train_%j.out
#SBATCH --mail-user=Sara.Navarro@stonybrook.edu

cd $HOME
source .bashrc
source modules.sh

conda activate xcq

cd /gpfs/scratch/sarnavarro/xcquinox/example_GGA_Sara/02.Training
python train_traj_Sara_GGA.py --train_traj_path ../../scripts/script_data/training_subsets/01/subat_ref.traj \
--train_data_dir ../../scripts/script_data/training_subsets/01 \
--xc_xc_net_path . \
--serial --xc_xc_level GGA \
--mf_grid_level 1 \
--n_steps 100 \
--singles_start 1
