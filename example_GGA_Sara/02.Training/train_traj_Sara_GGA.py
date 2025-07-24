    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('xyz', action='store', help ='Path to .xyz/.traj file containing list of configurations')
    parser.add_argument('-charge', '-c', action='store', type=int, help='Net charge of the system', default=0)
    parser.add_argument('-fdf', metavar='fdf', type=str, nargs = '?', default='')
    parser.add_argument('-basis', metavar='basis', type=str, nargs = '?', default='6-311++G(3df,2pd)', help='basis to use. default 6-311++G(3df,2pd)')
    parser.add_argument('-nworkers', metavar='nworkers', type=int, nargs = '?', default=1)
    parser.add_argument('-cmargin', '-cm', type=float, default=10.0, help='the margin to define extent of cube files generated')
    parser.add_argument('-xc', '--xc', type=str, default='pbe', help='Type of XC calculation. Either pbe or ccsdt.')
    parser.add_argument('-pbc', '--pbc', type=bool, default=False, help='Whether to do PBC calculation or not.')
    parser.add_argument('-L', '--L', type=float, default=None, help='The box length for the PBC cell')
    parser.add_argument('-df', '--df', type=bool, default=False, help='Choose whether or not the DFT calculation uses density fitting')
    parser.add_argument('-ps', '--pseudo', type=str, default=None, help='Pseudopotential choice. Currently either none or pbe')
    parser.add_argument('-r', '--rerun', type=bool, default=False, help='whether or not to continue and skip previously completed calculations or redo all')
    parser.add_argument('--ghf', default=False, action="store_true", help='whether to have wrapper guess HF to use or do GHF. if flagged, ')
    parser.add_argument('--serial', default=False, action="store_true", help="Run in serial, without DASK.")
    parser.add_argument('--overwrite_gcharge', default=False, action="store_true", help='Whether to try to overwrite specified CHARGE -c if atom.info has charge.')
    parser.add_argument('--restart', default=False, action="store_true", help='If flagged, will use checkfile as init guess for calculations.')
    parser.add_argument('--forcepol', default=False, action='store_true', help='If flagged, all calculations are spin polarized.')
    parser.add_argument('--testgen', default=False, action='store_true', help='If flagged, calculation stops after mol generation.')
    parser.add_argument('--startind', default=-1, type=int, action='store', help='SERIAL MODE ONLY. If specified, will skip indices in trajectory before given value')
    parser.add_argument('--endind', default=999999999999, type=int, action='store', help='SERIAL MODE ONLY. If specified, will only calculate up to this index')
    parser.add_argument('--atomize', default=False, action='store_true', help='If flagged, will generate predictions for the single-atom components present in trajectory molecules.')
    parser.add_argument('--mf_grid_level', type=int, default=3, action='store', help='Grid level for PySCF(AD) calculation')
    #add arguments for pyscfad network driver
    parser.add_argument('--xc_x_net_path', type=str, default='', action='store', help='Path to the trained xcquinox exchange network to use in PySCF(AD) as calculation driver\nParent directory of network assumed to be of form TYPE_MLPDEPTH_NHIDDEN_LEVEL (e.g. x_3_16_mgga)')
    parser.add_argument('--xc_x_ninput', type=int, action='store', help='Number of inputs the exchange network expects')
    parser.add_argument('--xc_x_use', nargs='+', type=int, action='store', default=[], help='Specify the desired indices for the exchange network to actually use, if not the full range of descriptors.')
    parser.add_argument('--xc_c_net_path', type=str, default='', action='store', help='Path to the trained xcquinox correlation network to use in PySCF(AD) as calculation driver\nParent directory of network assumed to be of form TYPE_MLPDEPTH_NHIDDEN_LEVEL (e.g. c_3_16_mgga)')
    parser.add_argument('--xc_c_ninput', type=int, action='store', help='Number of inputs the correlation network expects')
    parser.add_argument('--xc_c_use', nargs='+', type=int, action='store', default=[], help='Specify the desired indices for the correlation network to actually use, if not the full range of descriptors.')
    parser.add_argument('--xc_xc_net_path', type=str, default='', action='store', help='Path to the trained xcquinox exchange-correlation network to use in PySCF(AD) as calculation driver\nParent directory of network assumed to be of form TYPE_MLPDEPTH_NHIDDEN_LEVEL (e.g. xc_3_16_mgga)')
    parser.add_argument('--xc_xc_level', type=str, default='MGGA', action='store', choices=['GGA','MGGA','NONLOCAL','NL'], help='Type of network being loaded (GGA,MGGA,NONLOCAL,NL)')
    parser.add_argument('--xc_xc_ninput', type=int, action='store', help='Number of inputs the exchange-correlation network expects')
    parser.add_argument('--xc_verbose', default=False, action='store_true', help='If flagged, sets verbosity on the network.')
    parser.add_argument('--xc_full', default=False, action='store_true', help='If flagged, will deserialize a saved XC object, as opposed to individual X/C MLP networks.')
    #add arguments for training
    parser.add_argument('--n_steps', action='store', type=int, default=200, help='The number training epochs to go through.')
    parser.add_argument('--singles_start', action='store', type=int, default=4, help='The index at which the single-atom molecules start, assuming they are at the end of a list of trajectories containing molecules and reactions.')
    parser.add_argument('--do_jit', action='store_true', default=False, help='If flagged, will try to utilize JAX-jitting during training')
    parser.add_argument('--train_traj_path', action='store', type=str, default='/home/awills/Documents/Research/xcquinox/scripts/script_data/haunschild_g2/g2_97.traj', help='Location of the data file for use during pre-training')
    parser.add_argument('--train_data_dir', action='store', type=str, default='/home/awills/Documents/Research/xcquinox/scripts/script_data/haunschild_g2/g2_97.traj', help='Location of the data file for use during pre-training')
    parser.add_argument('--debug', action='store_true', help='If flagged, only selects first in the target list for quick debugging purposes.')
    parser.add_argument('--verbose', action='store_true', help='If flagged, activates verbosity flag in the network.')
    parser.add_argument('--init_lr', action='store', type=float, default=5e-2, help='The initial learning rate for the network.')
    parser.add_argument('--lr_decay_start', action='store', type=int, default=50, help='The epoch at which the exponential decay begins.')
    parser.add_argument('--lr_decay_rate', action='store', type=float, default=0.9, help='The decay rate for the exponential decay.')
    
    args = parser.parse_args()
    setattr(__config__, 'cubegen_box_margin', args.cmargin)
    GCHARGE = args.charge

    atoms = read(args.train_traj_path, ':')
    print("==================")
    print("ARGS SUMMARY")
    print(args)
    print("==================")
    
    ATOMIZATION = args.atomize
    if ATOMIZATION:
        print('ATOMZATION CALCULATION FLAGGED -- GETTING ATOMIC CONSTITUENTS.')
        total_syms = []
        for idx, at in enumerate(atoms):
            total_syms += at.get_chemical_symbols()
        total_syms = sorted(list(set(total_syms)))
        print('SINGLE ATOMS REPRESENTED: {}'.format(total_syms))
        singles = []
        for atm_idx, symbol in enumerate(total_syms):
            print('========================================================')
            print('GROUP {}/{} -- {}'.format(atm_idx, len(total_syms)-1, symbol))
            print('--------------------------------------------------------')

            molsym = symbol
            syms = [symbol]
            pos = [[0,0,0]]
            form = symbol
            spin = spins_dict[symbol]

            singles.append(Atoms(form, pos))
        print(f'Singles = {singles}')
        atoms = singles + atoms
    
    gridmodels = []
    CUSTOM_XC = False
    xcnet = None
    if args.xc_x_net_path:
        'xcquinox network exchange path provided, attempting read-in...'
        xnet, xlevel, loadnet = loadnet_from_strucdir(args.xc_x_net_path, args.xc_x_ninput, args.xc_x_use, xc_full=args.xc_full)
        gridmodels.append(xnet)
        CUSTOM_XC = True
    if args.xc_c_net_path:
        'xcquinox network exchange path provided, attempting read-in...'
        cnet, clevel, loadnet = loadnet_from_strucdir(args.xc_c_net_path, args.xc_c_ninput, args.xc_c_use, xc_full=args.xc_full)
        gridmodels.append(cnet)
        CUSTOM_XC = True
    if args.xc_xc_net_path:
        print('xcquinox network exchange-correlation path provided, attempting read-in...')
        if 'xc.eqx' in args.xc_xc_net_path:
            xcdsf = args.xc_xc_net_path.split('/')[-1]
            netpath = '/'.join(args.xc_xc_net_path.split('/')[:-1])
            print('Loading in xc checkpoint from: netpath={}, file={}'.format(netpath, xcdsf))
        else:
            xcdsf = ''
            netpath = args.xc_xc_net_path
            print('Loading in xc checkpoint from: netpath={}, file={}'.format(netpath, xcdsf))
        xcnet = xce.xc.get_xcfunc(args.xc_xc_level,
                               netpath, xcdsfile = xcdsf)
        CUSTOM_XC = True
    if args.xc_full and args.xc_xc_net_path:
        assert args.xc_x_net_path, "Must specify location of exchange network to generate template correctly"
        assert args.xc_c_net_path, "Must specify location of correlation network to generate template correctly"
        print('Full XC Object specified -- loading into constructed network')
        print('Specified path: {}'.format(args.xc_xc_net_path))
        xcnet = eqx.tree_deserialise_leaves(args.xc_xc_net_path, xcnet)
        print(xcnet)
        CUSTOM_XC = True

    input_xc = args.xc if not CUSTOM_XC else 'custom_xc'

    class E_DM_Loss(eqx.Module):
        """ Loss function for the XC model.
        Calculates the loss based on atomization energies and density matrices.

        Useful info:
         - All the dicto

        """

        def __call__(self, model, refdatatraj, refdatadir, singles_start):
            atoms = read(refdatatraj, ':')
            results = [do_net_calc(0, ia, atoms[ia], basis=args.basis,
                                            margin=args.cmargin,
                                            XC=input_xc,
                                            PBC=args.pbc,
                                            L=args.L,
                                            df=args.df,
                                            pseudo=args.pseudo,
                                            rerun=args.rerun,
                                            owcharge=args.overwrite_gcharge,
                                            restart = args.restart,
                                            forcepol = False,
                                            testgen = args.testgen,
                                            gridlevel = args.mf_grid_level,
                                            atomize = True,
                                            custom_xc = CUSTOM_XC,
                                            custom_xc_net = model) for ia in range(len(atoms)) if ia >= args.startind and ia < args.endind]
            e_dct = {str(idx)+'_'+str(res[0].symbols): res[1] for idx, res in enumerate(results)}
            dm_dct = {str(idx)+'_'+str(res[0].symbols): res[2] for idx, res in enumerate(results)}
            single_e_dct = {k.split('_')[1]:e_dct[k] for k in e_dct.keys() if int(k.split('_')[0]) >= singles_start and k.split('_')[1] in spins_dict.keys()}
            ae_e_dct = {str(idx)+'_'+str(res[0].symbols): res[1] for idx, res in enumerate(results) if not res[0].info.get('reaction', None) and str(res[0].symbols) not in single_e_dct.keys()}
            rxn_e_dct = {str(idx)+'_'+str(res[0].symbols): res[1] for idx, res in enumerate(results) if type(res[0].info.get('reaction', None)) == int}
            rxn_part_dct = {}
            for k in rxn_e_dct.keys():
                kidx, ksym = k.split('_')
                kidx = int(kidx)
                ncomps = results[kidx][0].info.get('reaction')
                rxncomps = [str(kidx-i)+'_'+str(results[kidx-i][0].symbols) for i in range(1,ncomps+1)]                    
                rxn_part_dct[k] = rxncomps
            print('------------------------------------------')
            print('DICTIONARY SUMMARIES: ')
            print(f'e_dct = {e_dct}')
            print(f'single_e_dct = {single_e_dct}')
            print(f'ae_e_dct = {ae_e_dct}')
            print(f'rxn_e_dct = {rxn_e_dct}')
            print(f'rxn_part_dct = {rxn_part_dct}')
            print('------------------------------------------')
            ae_losses = []
            dm_losses = []
            ae_losses = jnp.array(ae_losses)
            dm_losses = jnp.array(dm_losses)

            # ATOMIZATION ENERGY LOSS CALCULATION
            for k, v in ae_e_dct.items():
                kidx, at = k.split('_')
                kidx = int(kidx)
                ATE = v
                REFATE = atoms[kidx].info['target_energy']
                syms = atoms[kidx].get_chemical_symbols()
                print(f'{kidx}_{at} energy : {v}')
                for sym in syms:
                    ATE -= single_e_dct[sym]
                print(f'{kidx}_{at} atomization energy : {ATE}')
                print(f'{kidx}_{at} reference atomization energy : {REFATE}')
                ae_losses = jnp.append(ae_losses, ATE-REFATE)
            for k, v in rxn_e_dct.items():
                print(f'Reaction -- {k} : {rxn_part_dct[k]}')
                rxnloss = v
                for subcomp in rxn_part_dct[k]:
                    rxnloss -= e_dct[subcomp]
                print(f'Reaction -- {k} : Loss = {rxnloss}')
                ae_losses = jnp.append(ae_losses, rxnloss)
            for k, v in dm_dct.items():
                print('Density matrix losses: {}'.format(k))
                refdm = np.load(os.path.join(refdatadir, k+'.dm.npy'))
                refdm = jnp.array(refdm)
                print(f'{k}: Reference density matrix shape = {refdm.shape}')
                print(f'{k}: Predicted density matrix shape = {v.shape}')
                if refdm.shape != v.shape:
                    print('Reference and predicted density matrix shape mismatch. Reducing spin channels of non-matching.')
                    if len(refdm.shape) == 3:
                        refdm = refdm[0] + refdm[1]
                        print('New reference density matrix shape: {}'.format(refdm.shape))
                    if len(v.shape) == 3:
                        v = v[0] + v[1]
                        print('New predicted density matrix shape: {}'.format(v.shape))
                dm_losses = jnp.append(dm_losses, jnp.sum((refdm-v)**2))
            print('AE Losses: {}'.format(ae_losses))
            print('DM Losses: {}'.format(dm_losses))
            return jnp.sqrt(jnp.mean((ae_losses**2))) + jnp.sqrt(jnp.mean((dm_losses**2)))

    if not args.rerun:
        print('beginning new progress file')
        with open('progress','w') as progfile:
            progfile.write('#idx\tatoms.symbols\tetot  (Har)\tehf  (Har)\teccsd  (Har)\teccsdt  (Har)\n')
        print('beginning new nonconverged file')
        with open('unconv','w') as ucfile:
            ucfile.write('#idx\tatoms.symbols\tetot  (Har)\tehf  (Har)\teccsd  (Har)\teccsdt  (Har)\n')
        print('beginning new timing file')
        with open('timing','w') as tfile:
            tfile.write('#idx\tatoms.symbols\tcalc\ttime (s)\n')
        print("SERIAL CALCULATIONS")


        scheduler = optax.exponential_decay(init_value = args.init_lr, transition_begin=args.lr_decay_start, transition_steps=args.n_steps-args.lr_decay_start, decay_rate=args.lr_decay_rate)
        optimizer = optax.adam(learning_rate = scheduler)

        trainer = xce.train.xcTrainer(model=xcnet, optim=optimizer, steps=args.n_steps, loss = E_DM_Loss(), do_jit=args.do_jit, logfile='trlog')

        cpus = jax.devices(backend='cpu')
        with jax.default_device(cpus[0]):
            newm = trainer(1, trainer.model, [args.train_traj_path], [args.train_data_dir], [args.singles_start])
