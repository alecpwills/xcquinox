python makexcs_GGA.py \
  --outpath . \
  --load_xnet_path ../01.Pretraining/GGA_FcNet_sigma_d3_n16_s42_v_10000 \
  --load_cnet_path ../01.Pretraining/GGA_FcNet_sigma_d3_n16_s42_v_10000 \
  >> run_makexcs_GGA_Sigma.log 2>&1
echo "Finished running makexcs_GGA.py for GGA Sigma"
