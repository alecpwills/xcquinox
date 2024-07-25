python ~/Documents/Research/xcquinox/scripts/pretrain_exc2.py \
--pretrain_level GGA --init_lr 0.005 \
--pretrain_net_path /home/awills/Documents/Research/xcquinox/scripts/script_data/ctests/pt/pbe2/xc_3_16_c0_gga \
--pretrain_xc PBE \
--n_steps 1000 \
--g297_data_path ~/Documents/Research/xcquinox/scripts/script_data/haunschild_g2/g2_97.traj \
--pretrain_net xc --full_calc \
--ref_calc_dir /home/awills/Documents/Research/xcquinox/scripts/script_data/haunschild_g2/evaluations/pbe_xc
