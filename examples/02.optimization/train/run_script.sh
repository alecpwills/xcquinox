python ../../../scripts/train_traj.py --train_traj_path ../../../scripts/script_data/training_subsets/01/subat_ref.traj \
--train_data_dir ../../../scripts/script_data/training_subsets/01 \
--xc_xc_net_path ../mgga/xc_3_16_0_mgga/ \
--serial --xc_xc_level MGGA \
--mf_grid_level 1 \
--n_steps 100 \
--singles_start 1
