python ~/Documents/Research/xcquinox/scripts/pretrain_exc2.py \
--pretrain_level GGA \
--pretrain_net_path /home/awills/Documents/Research/xcquinox/scripts/script_data/ctests/pt/pbe2/xc_3_16_c0_gga \
--pretrain_xc PBE \
--n_steps 100 \
--g297_data_path ~/Documents/Research/xcquinox/scripts/script_data/haunschild_g2/g2_97.traj \
--pretrain_net xc --do_jit
