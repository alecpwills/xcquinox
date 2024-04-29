#!/usr/bin/bash
pypatch apply pyscfad.dft.patch pyscfad.dft
pypatch apply pyscfad.scf.patch pyscfad.scf
cp ./pyscf_conf.py ~/.pyscf_conf.py